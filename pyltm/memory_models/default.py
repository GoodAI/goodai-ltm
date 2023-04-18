import gc
from typing import List, Union, Any, Callable, Set, Optional
import numpy as np
import torch
from faiss import Index
from transformers import PreTrainedTokenizer

from pyltm.embeddings import BaseTextEmbeddingModel
from pyltm.helpers.tokenizer_helper import get_sentence_punctuation_ids, get_pad_token_id
from pyltm.matching import BaseTextMatchingModel
from pyltm.memory_models.chunk_queue import ChunkQueue, BaseChunkQueue
from pyltm.memory_models.config import TextMemoryConfig
from pyltm.memory_models.mem_foundation import BaseTextMemoryFoundation
from pyltm.memory_models.simple_vector_db import SimpleVectorDb

_vector_db_type = Union[Index, SimpleVectorDb]


class DefaultTextMemory(BaseTextMemoryFoundation):
    def __init__(self, vector_db: _vector_db_type, tokenizer: PreTrainedTokenizer,
                 emb_model: BaseTextEmbeddingModel, matching_model: Optional[BaseTextMatchingModel],
                 device: torch.device, config: TextMemoryConfig,
                 chunk_queue_fn: Callable[[], BaseChunkQueue] = None):
        if chunk_queue_fn is None:
            def chunk_queue_fn():
                return ChunkQueue(config.queue_capacity, config.chunk_capacity)

        self.config = config
        self.device = device
        self.emb_model = emb_model
        self.matching_model = matching_model
        self.em_tokenizer = tokenizer
        self.bucket_capacity = config.chunk_capacity
        self.pad_token_id = get_pad_token_id(self.em_tokenizer)
        self.punctuation_ids = get_sentence_punctuation_ids(self.em_tokenizer, include_line_break=False)
        self.chunk_queue: BaseChunkQueue = chunk_queue_fn()
        self.vector_db = vector_db
        has_matching_model = self.matching_model is not None
        super().__init__(vector_db, self.em_tokenizer, has_matching_model,
                         self.emb_model.get_num_storage_embeddings(), device, config.adjacent_chunks_ok)

    def get_tokenizer(self):
        return self.em_tokenizer

    def get_queue_capacity(self):
        return self.chunk_queue.get_capacity()

    def set_queue_capacity(self, num_buckets: int):
        self.chunk_queue.capacity = num_buckets

    def get_metadata(self, chunk_id: int) -> Optional[Any]:
        chunk = self.chunk_queue.get_chunk(chunk_id)
        return chunk.metadata

    def retrieve_chunk_sequences(self, chunk_ids: List[int]):
        return self.chunk_queue.retrieve_chunk_sequences(chunk_ids)

    def retrieve_complete_sequences(self, chunk_ids: List[int], punctuation_ids: Set[int]):
        return self.chunk_queue.retrieve_complete_sequences(chunk_ids, punctuation_ids)

    def get_retrieval_key_for_text(self, queries: List[str]) -> torch.Tensor:
        return self.emb_model.encode_queries(queries, convert_to_tensor=True)

    def get_match_probabilities(self, query: str, passages: List[str]) -> List[float]:
        if self.matching_model is None:
            raise SystemError('No query-passage match probability model available')
        return self.matching_model.get_match_confidence(query, passages)

    def add_text(self, text: str, metadata: Optional[Any]):
        token_ids = self.em_tokenizer.encode(text, add_special_tokens=False)
        removed_buckets = self.chunk_queue.add_sequence(token_ids, metadata)
        self._ensure_keys_added()
        removed_indexes = [rb.index for rb in removed_buckets]
        if len(removed_indexes) > 0:
            self.vector_db.remove_ids(np.array(removed_indexes).astype(np.int64))

    def retrieve_all_text(self) -> str:
        token_ids = self.chunk_queue.get_latest_token_ids(max_num_tokens=None)
        return self.em_tokenizer.decode(token_ids, skip_special_tokens=True)

    def retrieve_all_chunks(self) -> List[str]:
        chunk_list = self.chunk_queue.get_chunk_sequences()
        tok = self.em_tokenizer
        return [tok.decode(seq, skip_special_tokens=True) for seq in chunk_list]

    def _ensure_keys_added(self, batch_size=50):
        picked_buckets, token_id_matrix = self.chunk_queue.get_chunks_for_indexing()
        emb_model = self.emb_model
        sk_list = []
        num_sequences = len(token_id_matrix)
        for i in range(0, num_sequences, batch_size):
            token_id_batch = token_id_matrix[i:i + batch_size]
            text_batch = self.em_tokenizer.batch_decode(token_id_batch, skip_special_tokens=True)
            sk_batch = emb_model.encode_corpus(text_batch, convert_to_tensor=True)
            sk_list.append(sk_batch.detach())
            if num_sequences > batch_size:
                del sk_batch
                gc.collect()
        if len(sk_list) > 0:
            sk_all = torch.cat(sk_list)
            num_chunks, num_sk = sk_all.size(0), sk_all.size(1),
            sk_all = sk_all.view(num_chunks * num_sk, -1)
            sk_all_np = sk_all.cpu().numpy().astype(np.float32)
            b_indexes = []
            for bucket in picked_buckets:
                if bucket.is_at_capacity():
                    bucket.set_indexed(True)
                b_indexes.extend([bucket.index] * num_sk)
            b_indexes_np = np.array(b_indexes).astype(np.int64)
            self.vector_db.add_with_ids(sk_all_np, b_indexes_np)

    def clear(self):
        self.chunk_queue.flush()
        self.vector_db.reset()

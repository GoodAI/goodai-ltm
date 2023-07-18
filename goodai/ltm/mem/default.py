import math
from typing import List, Union, Any, Optional, Tuple, Callable
import numpy as np
import torch
from faiss import Index
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from goodai.helpers.tokenizer_helper import get_pad_token_id, get_sentence_punctuation_ids
from goodai.ltm.embeddings.base import BaseTextEmbeddingModel
from goodai.ltm.mem.base import BaseReranker, BaseImportanceModel
from goodai.ltm.mem.chunk import Chunk, TextKeyType
from goodai.ltm.mem.rewrite_model import BaseRewriteModel
from goodai.ltm.reranking.base import BaseTextMatchingModel
from goodai.ltm.mem.chunk_queue import ChunkQueue, PassageInfo, ChunkExpansionOptions
from goodai.ltm.mem.config import TextMemoryConfig
from goodai.ltm.mem.mem_foundation import BaseTextMemoryFoundation, VectorDbType
from goodai.ltm.mem.simple_vector_db import SimpleVectorDb

_vector_db_type = Union[Index, SimpleVectorDb]


class DefaultTextMemory(BaseTextMemoryFoundation):
    def __init__(self, vector_db_type: VectorDbType, tokenizer: PreTrainedTokenizer,
                 emb_model: BaseTextEmbeddingModel, matching_model: Optional[BaseTextMatchingModel],
                 device: torch.device, config: TextMemoryConfig,
                 query_rewrite_model: Optional[BaseRewriteModel] = None,
                 memory_rewrite_model: Optional[BaseRewriteModel] = None,
                 reranker: Optional[BaseReranker] = None,
                 importance_model: Optional[BaseImportanceModel] = None,
                 ):
        cc = config.chunk_capacity
        cof = config.chunk_overlap_fraction
        if cof < 0 or cof > 0.5:
            raise ValueError(f'Invalid chunk overlap fraction: {cof}')
        ciao = cc - math.ceil(cc * cof)
        self.chunk_queue = ChunkQueue(config.queue_capacity, cc, ciao)
        self.config = config
        self.device = device
        self.emb_model = emb_model
        self.matching_model = matching_model
        self.chunk_tokenizer = tokenizer
        self.bucket_capacity = config.chunk_capacity
        self.pad_token_id = get_pad_token_id(self.chunk_tokenizer)
        self.punctuation_ids = get_sentence_punctuation_ids(self.chunk_tokenizer, include_line_break=False)
        self.query_rewrite_model = query_rewrite_model
        self.memory_rewrite_model = memory_rewrite_model
        self.importance_model = importance_model
        self.reranker = reranker
        self.ce_options = ChunkExpansionOptions.from_config(tokenizer, config.chunk_expansion_config)
        has_matching_model = self.matching_model is not None
        super().__init__(vector_db_type, self.chunk_tokenizer, has_matching_model,
                         self.emb_model.get_num_storage_embeddings(),
                         self.emb_model.get_embedding_dim(),
                         config.chunk_capacity,
                         config.reranking_k_factor, config.max_query_length,
                         query_rewrite_model, reranker,
                         config.chunk_overlap_fraction, config.redundancy_overlap_threshold,
                         self.ce_options, device)

    def has_importance_model(self) -> bool:
        return self.importance_model is not None

    def get_tokenizer(self):
        return self.chunk_tokenizer

    def get_queue_capacity(self):
        return self.chunk_queue.get_capacity()

    def set_queue_capacity(self, num_buckets: int):
        self.chunk_queue.capacity = num_buckets

    def get_metadata(self, chunk_id: int) -> Optional[Any]:
        chunk = self.chunk_queue.get_chunk(chunk_id)
        if chunk is None:
            return None
        return chunk.metadata

    def get_chunk_text(self, chunk: Chunk) -> str:
        token_ids = self.chunk_queue.get_chunk_token_ids(chunk)
        return self.chunk_tokenizer.decode(token_ids, skip_special_tokens=True)

    def get_complete_passage(self, chunk: Chunk) -> PassageInfo:
        return self.chunk_queue.get_complete_passage(chunk, self.ce_options)

    def get_chunk(self, chunk_id: int) -> Chunk:
        return self.chunk_queue.get_chunk(chunk_id)

    def retrieve_chunk_sequences(self, chunks: List[Chunk]):
        return self.chunk_queue.retrieve_chunk_sequences_given_chunks(chunks)

    def get_text(self, text_key: TextKeyType) -> Optional[str]:
        token_ids = self.chunk_queue.get_sequence_token_ids(text_key)
        if token_ids is None:
            return None
        return self.chunk_tokenizer.decode(token_ids, skip_special_tokens=True)

    def get_retrieval_key_for_text(self, queries: List[str], show_progress_bar: bool = False) -> torch.Tensor:
        return self.emb_model.encode_queries(queries, convert_to_tensor=True, show_progress_bar=show_progress_bar)

    def predict_match(self, sentences: List[Tuple[str, str]], show_progress_bar: bool = False,
                      batch_size: int = 32) -> List[float]:
        return self.matching_model.predict(sentences, show_progress_bar=show_progress_bar,
                                           batch_size=batch_size)

    def _replace_or_add_text(self, cq_fn: Callable, text: str, metadata: Optional[Any] = None, rewrite: bool = False,
                             rewrite_context: Optional[str] = None, show_progress_bar: bool = False,
                             timestamp: Optional[float] = None, text_key: TextKeyType = None):
        if rewrite and not self.memory_rewrite_model:
            raise ValueError("For memory rewriting, a rewriting model must be provided")
        if rewrite and self.memory_rewrite_model:
            text = self.memory_rewrite_model.rewrite_memory(text, rewrite_context)
        importance = None
        if self.importance_model:
            importance = self.importance_model.get_importance(text)
        token_ids = self.chunk_tokenizer.encode(text, add_special_tokens=False)
        cq_params = dict(
            new_token_ids=token_ids,
            metadata=metadata,
            importance=importance,
            timestamp=timestamp,
        )
        if text_key is not None:
            cq_params['text_key'] = text_key
        removed_chunks, text_key = cq_fn(**cq_params)
        self._ensure_keys_added(show_progress_bar=show_progress_bar)
        removed_chunk_ids = [rb.chunk_id for rb in removed_chunks]
        if len(removed_chunk_ids) > 0:
            self.vector_db.remove_ids(np.array(removed_chunk_ids).astype(np.int64))
        return text_key

    def add_text(self, text: str, metadata: Optional[dict] = None, rewrite: bool = False,
                 rewrite_context: Optional[str] = None, show_progress_bar: bool = False,
                 timestamp: Optional[float] = None) -> TextKeyType:
        return self._replace_or_add_text(self.chunk_queue.add_sequence,
                                         text=text, metadata=metadata, rewrite=rewrite,
                                         rewrite_context=rewrite_context, show_progress_bar=show_progress_bar,
                                         timestamp=timestamp)

    def replace_text(self, text_key: TextKeyType, text: str, metadata: Optional[dict] = None,
                     rewrite: bool = False, rewrite_context: Optional[str] = None,
                     show_progress_bar: bool = False, timestamp: Optional[float] = None) -> TextKeyType:
        return self._replace_or_add_text(self.chunk_queue.replace_sequence,
                                         text=text, metadata=metadata, rewrite=rewrite,
                                         rewrite_context=rewrite_context, show_progress_bar=show_progress_bar,
                                         timestamp=timestamp, text_key=text_key)

    def delete_text(self, text_key: TextKeyType, show_progress_bar: bool = False) -> TextKeyType:
        return self.replace_text(text_key, "", show_progress_bar=show_progress_bar)

    def add_separator(self):
        self.chunk_queue.add_separator()

    def is_empty(self) -> bool:
        return len(self.chunk_queue.token_ids) == 0

    def retrieve_all_text(self) -> str:
        token_ids = self.chunk_queue.get_latest_token_ids(max_num_tokens=None)
        return self.chunk_tokenizer.decode(token_ids, skip_special_tokens=True)

    def retrieve_all_chunks(self) -> List[str]:
        chunk_list = self.chunk_queue.get_chunk_sequences()
        tok = self.chunk_tokenizer
        return [tok.decode(seq, skip_special_tokens=True) for seq in chunk_list]

    def get_all_chunks(self) -> List[Chunk]:
        return self.chunk_queue.get_all_chunks()

    def _ensure_keys_added(self, batch_size=50, show_progress_bar: bool = False):
        picked_chunks, token_id_matrix = self.chunk_queue.get_chunks_for_indexing()
        emb_model = self.emb_model
        sk_list = []
        num_sequences = len(token_id_matrix)
        rng = range(0, num_sequences, batch_size)
        if show_progress_bar:
            rng = tqdm(rng, desc='Storage', unit='batch')
        for i in rng:
            token_id_batch = token_id_matrix[i:i + batch_size]
            text_batch = self.chunk_tokenizer.batch_decode(token_id_batch, skip_special_tokens=True)
            sk_batch = emb_model.encode_corpus(text_batch, convert_to_tensor=True,
                                               show_progress_bar=False, batch_size=batch_size)
            if sk_batch.size(0) != len(text_batch):
                raise RuntimeError(f'Number of storage embeddings returned by embedding model is {sk_batch.size(0)}, '
                                   f'while the number of encoded texts is {len(text_batch)}')
            sk_list.append(sk_batch.detach())
        if len(sk_list) > 0:
            sk_all = torch.cat(sk_list)
            num_chunks, num_sk = sk_all.size(0), sk_all.size(1),
            sk_all = sk_all.view(num_chunks * num_sk, -1)
            sk_all_np = sk_all.cpu().numpy().astype(np.float32)
            b_indexes = []
            for chunk in picked_chunks:
                chunk.update_indexed_state()
                b_indexes.extend([chunk.chunk_id] * num_sk)
            b_indexes_np = np.array(b_indexes).astype(np.int64)
            self.vector_db.add_with_ids(sk_all_np, b_indexes_np)

    def clear(self):
        self.chunk_queue.flush()
        self.vector_db.reset()

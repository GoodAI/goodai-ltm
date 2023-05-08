import abc
import enum
import io
import json
import sys
from typing import List, Union, Set, Optional, Tuple

import faiss
import numpy as np
import torch
from transformers import PreTrainedTokenizer

from goodai.helpers.collections_helper import remove_duplicates, num_visited_to_get_expected_count, \
    get_non_adjacent
from goodai.helpers.tokenizer_helper import get_sentence_punctuation_ids
from goodai.ltm.mem.base import BaseTextMemory, RetrievedMemory
from goodai.ltm.mem.chunk import Chunk
from goodai.ltm.mem.simple_vector_db import SimpleVectorDb

_vector_db_type = Union[faiss.Index, SimpleVectorDb]


class VectorDbType(enum.Enum):
    SIMPLE = 0,
    FAISS_FLAT_L2 = 1,


class BaseTextMemoryFoundation(BaseTextMemory):
    def __init__(self, vector_db_type: VectorDbType, tokenizer: PreTrainedTokenizer, has_match_prob_model: bool,
                 num_storage_embeddings: int, emb_dim: int, reranking_k_multiplier: int,
                 device: torch.device):
        super().__init__()
        self.reranking_k_multiplier = reranking_k_multiplier
        self.num_storage_embeddings = num_storage_embeddings
        self.has_match_prob_model = has_match_prob_model
        self.vector_db = self.create_vector_db(vector_db_type, emb_dim)
        self.adjacent_chunks_ok = False  # TODO should be dealt with after chunking configuration enhancements
        self.device = device
        self.chunk_tokenizer = tokenizer
        self.punctuation_ids = get_sentence_punctuation_ids(self.chunk_tokenizer, include_line_break=False)

    @staticmethod
    def create_vector_db(vector_db_type: VectorDbType, emb_dim: int) -> _vector_db_type:
        if vector_db_type == VectorDbType.SIMPLE:
            return SimpleVectorDb()
        elif vector_db_type == VectorDbType.FAISS_FLAT_L2:
            return faiss.IndexIDMap(faiss.IndexFlatL2(emb_dim))
        else:
            raise ValueError(f'Unrecognized vector DB type: {vector_db_type}')

    def get_tokenizer(self):
        return self.chunk_tokenizer

    @abc.abstractmethod
    def retrieve_chunk_sequences(self, chunk_ids: List[int]):
        pass

    @abc.abstractmethod
    def retrieve_complete_sequences(self, chunk_ids: List[int], punctuation_ids: Set[int]):
        pass

    @abc.abstractmethod
    def get_retrieval_key_for_text(self, queries: List[str], show_progress_bar: bool = False) -> torch.Tensor:
        pass

    def predict_match(self, sentences: List[Tuple[str, str]], show_progress_bar: bool = False) -> List[float]:
        pass

    @abc.abstractmethod
    def get_metadata(self, chunk_id: int) -> Optional[dict]:
        pass

    @abc.abstractmethod
    def get_all_chunks(self) -> List[Chunk]:
        pass

    @abc.abstractmethod
    def get_chunk_text(self, chunk: Chunk) -> str:
        pass

    def dump(self, stream: io.TextIOBase = sys.stdout):
        chunks = self.get_all_chunks()
        stream.write('| Id | Metadata | Content |\n')
        stream.write('| ----- | -------- | ------- |\n')
        for chunk in chunks:
            chunk_text = self.get_chunk_text(chunk)
            ct_js = json.dumps(chunk_text)
            stream.write(f'| {chunk.index} | {chunk.metadata} | {ct_js} |')
            stream.write('\n')
        stream.flush()

    def _retrieve_for_scored_tuples(self, chunk_score_tuples: List[Tuple[float, tuple]], k: int) -> List[RetrievedMemory]:
        has_pm = self.has_match_prob_model
        chunk_score_tuples = chunk_score_tuples[:k]
        final_indexes = [ci for _, (_, ci) in chunk_score_tuples]
        sequences = self.retrieve_complete_sequences(final_indexes, self.punctuation_ids)
        retrieved_texts = self.chunk_tokenizer.batch_decode(sequences, skip_special_tokens=True)
        result = []
        for cs_tuple, r_text in zip(chunk_score_tuples, retrieved_texts):
            confidence, (distance, chunk_id) = cs_tuple
            metadata = self.get_metadata(chunk_id)
            if not has_pm:
                confidence = None
            result.append(RetrievedMemory(passage=r_text.strip(), distance=distance,
                                          confidence=confidence, metadata=metadata))
        return result

    def _multi_retrieve_for_tuples(self, queries: List[str], prelim_dist_indexes: List[List[Tuple[float, int]]],
                                   expected_key_db_top_k: int, k: int,
                                   show_progress_bar: bool = False) -> List[List[RetrievedMemory]]:
        adjacent_chunks_ok = self.adjacent_chunks_ok
        has_pm = self.has_match_prob_model
        # At this point:
        # - Duplicate chunk IDs have already been removed in prelim_dist_indexes
        # - prelim_dist_indexes is ordered by distance
        if has_pm:
            m_sentences = []
            m_indexes = []
            prelim_dist_indexes_list = []
            for i, (query, row_tuples) in enumerate(zip(queries, prelim_dist_indexes)):
                # Heuristic: Assuming embedding ordering is approximately good enough
                # to determine if adjacent chunks would be removed and should not be considered further
                nv = num_visited_to_get_expected_count(row_tuples, expected_key_db_top_k, adjacent_chunks_ok,
                                                       key_fn=lambda _t: _t[1])
                row_tuples = row_tuples[:nv]
                prelim_dist_indexes_list.append(row_tuples)
                prelim_chunk_indexes = [ci for _, ci in row_tuples]
                chunk_sequences: List[List[int]] = self.retrieve_chunk_sequences(prelim_chunk_indexes)
                chunk_texts = self.chunk_tokenizer.batch_decode(chunk_sequences, skip_special_tokens=True)
                for ct in chunk_texts:
                    m_sentences.append((query, ct,))
                    m_indexes.append(i)
            match_probs = self.predict_match(m_sentences, show_progress_bar=show_progress_bar)
            chunk_probs_list = [[] for _ in range(len(queries))]
            for m_index, match_prob in zip(m_indexes, match_probs):
                chunk_probs_list[m_index].append(match_prob)
            chunk_score_tuples_list = []
            for chunk_probs, row_tuples in zip(chunk_probs_list, prelim_dist_indexes_list):
                chunk_score_tuples = list(zip(chunk_probs, row_tuples))
                chunk_score_tuples.sort(key=lambda _t: _t[0], reverse=True)
                chunk_score_tuples_list.append(chunk_score_tuples)
        else:
            chunk_score_tuples_list = []
            for row_tuples in prelim_dist_indexes:
                pseudo_scores = list(range(len(row_tuples), 0, -1))
                chunk_score_tuples = list(zip(pseudo_scores, row_tuples))
                chunk_score_tuples_list.append(chunk_score_tuples)
        if not adjacent_chunks_ok:
            new_cstl = []
            for cst_entry in chunk_score_tuples_list:
                new_cst_entry = get_non_adjacent(cst_entry, key_fn=lambda _t: _t[1][1])
                new_cstl.append(new_cst_entry)
            chunk_score_tuples_list = new_cstl
        result = []
        for cst_entry in chunk_score_tuples_list:
            result.append(self._retrieve_for_scored_tuples(cst_entry, k=k))
        return result

    def _multi_retrieve(self, queries: List[str], distances: np.ndarray, indexes: np.ndarray,
                        expected_key_db_top_k: int, k: int,
                        show_progress_bar: bool = False) -> List[List[RetrievedMemory]]:
        # len(queries) == batch_size
        # distances, indexes: (batch_size, downstream_top_k)
        prelim_dist_indexes = []
        batch_size = len(queries)
        assert batch_size == distances.shape[0] == indexes.shape[0], \
            f'batch_size={batch_size}, distances.shape={distances.shape}, indexes.shape={indexes.shape}'
        for i in range(batch_size):
            row_d = distances[i]
            row_i = indexes[i]
            row_tuples = list(zip(row_d, row_i))
            row_tuples = remove_duplicates(row_tuples, key_fn=lambda _t: _t[1])
            prelim_dist_indexes.append(row_tuples)
        return self._multi_retrieve_for_tuples(queries, prelim_dist_indexes, expected_key_db_top_k, k,
                                               show_progress_bar=show_progress_bar)

    def _truncate_queries(self, queries: List[str], max_query_length: int):
        if max_query_length <= 0:
            raise ValueError(f'max_query_length={max_query_length} is invalid')
        ct = self.chunk_tokenizer
        tokenization = ct.batch_encode_plus(queries, add_special_tokens=False,
                                            return_attention_mask=False)
        token_ids: List[List[int]] = tokenization['input_ids']
        result = []
        for seq, query in zip(token_ids, queries):
            if len(seq) > max_query_length:
                truncated_seq = seq[-max_query_length:]
                result.append(ct.decode(truncated_seq, skip_special_tokens=True))
            else:
                result.append(query)
        return result

    def retrieve_multiple(self, queries: List[str], k: int, rewrite: bool = False, show_progress_bar: bool = False,
                          max_query_length: Optional[int] = 40,
                          **kwargs) -> List[List[RetrievedMemory]]:
        if max_query_length is not None:
            queries = self._truncate_queries(queries, max_query_length)
        rk = self.get_retrieval_key_for_text(queries, show_progress_bar=show_progress_bar)
        batch_size, num_rk, emb_size = rk.size(0), rk.size(1), rk.size(2),
        if num_rk != 1:
            raise ValueError('Memory does not support multiple retrieval embeddings')
        n_queries = len(queries)
        if batch_size != n_queries:
            raise SystemError(f'Batch returned by embeddings model is of shape {rk.shape} '
                              f'while the number of queries is {n_queries}')
        rk_np = rk.view(batch_size, emb_size).detach().cpu().numpy()
        adjacent_chunks_ok = self.adjacent_chunks_ok
        expected_key_db_top_k = k
        if self.has_match_prob_model:
            expected_key_db_top_k *= self.reranking_k_multiplier
        downstream_top_k = expected_key_db_top_k * self.num_storage_embeddings
        if not adjacent_chunks_ok:
            downstream_top_k *= 3
        distances, indexes = self.vector_db.search(rk_np, k=downstream_top_k)
        assert distances.shape[0] == indexes.shape[0] == batch_size
        return self._multi_retrieve(queries, distances, indexes, expected_key_db_top_k, k,
                                    show_progress_bar=show_progress_bar)

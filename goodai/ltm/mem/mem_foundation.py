import abc
import enum
import io
import json
import sys
from dataclasses import dataclass
from typing import List, Union, Set, Optional, Tuple

import faiss
import numpy as np
import torch
from transformers import PreTrainedTokenizer

from goodai.helpers.tokenizer_helper import get_sentence_punctuation_ids
from goodai.ltm.mem.base import BaseTextMemory, RetrievedMemory
from goodai.ltm.mem.chunk import Chunk
from goodai.ltm.mem.simple_vector_db import SimpleVectorDb

_vector_db_type = Union[faiss.Index, SimpleVectorDb]


class VectorDbType(enum.Enum):
    SIMPLE = 0,
    FAISS_FLAT_L2 = 1,


@dataclass
class RetrievedChunk:
    chunk: Chunk
    distance: float
    confidence: Optional[float] = None

    def sort_key(self):
        return self.distance if self.confidence is None else -self.confidence

    def with_confidence(self, confidence: float) -> 'RetrievedChunk':
        return RetrievedChunk(self.chunk, self.distance, confidence)

    @staticmethod
    def has_overlap(included_indexes: Set[int], chunk: Chunk,
                    overlap_threshold: float):
        chunk_from = chunk.from_token_seq_id
        chunk_to = chunk.to_token_seq_id
        if chunk_from not in included_indexes and (chunk_to - 1) not in included_indexes:
            return False
        if chunk_to <= chunk_from:
            return False
        intersection = included_indexes.intersection(range(chunk_from, chunk_to))
        len_inter = len(intersection)
        if len_inter <= 0:
            return False
        overlap_fraction = len_inter / (chunk_to - chunk_from)
        return overlap_fraction >= overlap_threshold

    @classmethod
    def remove_duplicates_and_overlaps(cls, items: List['RetrievedChunk'],
                                       overlap_threshold: float, max_count: int) -> List['RetrievedChunk']:
        id_set = set()
        result = []
        included_indexes: Set[int] = set()
        for item in items:
            chunk = item.chunk
            if chunk is None:
                continue
            chunk_id = chunk.chunk_id
            if chunk_id in id_set:
                continue
            id_set.add(chunk_id)
            if cls.has_overlap(included_indexes, chunk, overlap_threshold):
                continue
            result.append(item)
            if len(result) >= max_count:
                break
            chunk_from = chunk.from_token_seq_id
            chunk_to = chunk.to_token_seq_id
            included_indexes.update(range(chunk_from, chunk_to))
        return result


class BaseTextMemoryFoundation(BaseTextMemory):
    def __init__(self, vector_db_type: VectorDbType, tokenizer: PreTrainedTokenizer, has_match_prob_model: bool,
                 num_storage_embeddings: int, emb_dim: int,
                 chunk_capacity: int, reranking_k_factor: float,
                 max_query_length: Optional[int], overlap_fraction: float,
                 overlap_threshold: float,device: torch.device):
        super().__init__()
        if overlap_fraction < 0 or overlap_fraction > 0.5:
            raise ValueError(f'Invalid chunk overlap fraction: {overlap_fraction}')
        if overlap_threshold <= 0 or overlap_threshold > 1.0:
            raise ValueError(f'Invalid redundancy overlap threshold: {overlap_threshold}')
        self.chunk_capacity = chunk_capacity
        self.overlap_threshold = overlap_threshold
        self.overlap_fraction = overlap_fraction
        self.max_query_length = max_query_length
        self.reranking_k_factor = reranking_k_factor
        self.num_storage_embeddings = num_storage_embeddings
        self.has_match_prob_model = has_match_prob_model
        self.vector_db = self.create_vector_db(vector_db_type, emb_dim)
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
    def retrieve_chunk_sequences(self, chunks: List[Chunk]):
        pass

    @abc.abstractmethod
    def retrieve_complete_sequences(self, chunk_ids: List[int], punctuation_ids: Set[int]):
        pass

    @abc.abstractmethod
    def get_retrieval_key_for_text(self, queries: List[str], show_progress_bar: bool = False) -> torch.Tensor:
        pass

    def predict_match(self, sentences: List[Tuple[str, str]], show_progress_bar: bool = False,
                      batch_size: int = 32) -> List[float]:
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

    @abc.abstractmethod
    def get_chunk(self, chunk_id: int) -> Chunk:
        pass

    def dump(self, stream: io.TextIOBase = sys.stdout):
        chunks = self.get_all_chunks()
        stream.write('| Id | Metadata | Content |\n')
        stream.write('| ----- | -------- | ------- |\n')
        for chunk in chunks:
            chunk_text = self.get_chunk_text(chunk)
            ct_js = json.dumps(chunk_text)
            stream.write(f'| {chunk.chunk_id} | {chunk.metadata} | {ct_js} |')
            stream.write('\n')
        stream.flush()

    def _retrieve_for_processed_r_chunks(self, processed_r_chunks: List[RetrievedChunk], k: int) -> \
            List[RetrievedMemory]:
        has_pm = self.has_match_prob_model
        processed_r_chunks = processed_r_chunks[:k]
        final_chunk_ids: List[int] = [r_chunk.chunk.chunk_id for r_chunk in processed_r_chunks]
        sequences = self.retrieve_complete_sequences(final_chunk_ids, self.punctuation_ids)
        retrieved_texts = self.chunk_tokenizer.batch_decode(sequences, skip_special_tokens=True)
        result = []
        for r_chunk, r_text in zip(processed_r_chunks, retrieved_texts):
            confidence = r_chunk.confidence
            metadata = r_chunk.chunk.metadata
            if not has_pm:
                confidence = None
            result.append(RetrievedMemory(passage=r_text.strip(), distance=r_chunk.distance,
                                          confidence=confidence, metadata=metadata))
        return result

    def _multi_retrieve_for_r_chunks(self, queries: List[str], prelim_r_chunks: List[List[RetrievedChunk]], k: int,
                                     show_progress_bar: bool = False) -> List[List[RetrievedMemory]]:
        # At this point:
        # - No chunks are None
        # - Duplicate chunk IDs have already been removed in prelim_r_chunks
        # - Overlapping chunks have already been removed
        # - The length of prelim_r_chunks elements should be k or whatever is
        #   expected by the matching model, if any
        # - prelim_r_chunks is ordered by distance
        if self.has_match_prob_model:
            m_sentences = []
            m_indexes = []
            for i, (query, row_r_chunks) in enumerate(zip(queries, prelim_r_chunks)):
                prelim_chunks = [r_chunk.chunk for r_chunk in row_r_chunks]
                chunk_sequences: List[List[int]] = self.retrieve_chunk_sequences(prelim_chunks)
                chunk_texts = self.chunk_tokenizer.batch_decode(chunk_sequences, skip_special_tokens=True)
                for ct in chunk_texts:
                    m_sentences.append((query, ct,))
                    m_indexes.append(i)
            match_probs = self.predict_match(m_sentences, show_progress_bar=show_progress_bar)
            chunk_probs_list = [[] for _ in range(len(queries))]
            for m_index, match_prob in zip(m_indexes, match_probs):
                chunk_probs_list[m_index].append(match_prob)
            processed_r_chunks = []
            for chunk_probs, row_r_chunks in zip(chunk_probs_list, prelim_r_chunks):
                new_row_r_chunks = [r_chunk.with_confidence(cp) for cp, r_chunk in zip(chunk_probs, row_r_chunks)]
                new_row_r_chunks.sort(key=RetrievedChunk.sort_key)
                processed_r_chunks.append(new_row_r_chunks)
        else:
            processed_r_chunks = prelim_r_chunks
        result = []
        for row_r_chunks in processed_r_chunks:
            result.append(self._retrieve_for_processed_r_chunks(row_r_chunks, k=k))
        return result

    def _multi_retrieve(self, queries: List[str], distances: np.ndarray, indexes: np.ndarray,
                        expected_key_db_top_k: int, k: int,
                        show_progress_bar: bool = False) -> List[List[RetrievedMemory]]:
        # len(queries) == batch_size
        # distances, indexes: (batch_size, downstream_top_k)
        prelim_r_chunks = []
        batch_size = len(queries)
        assert batch_size == distances.shape[0] == indexes.shape[0], \
            f'batch_size={batch_size}, distances.shape={distances.shape}, indexes.shape={indexes.shape}'
        for i in range(batch_size):
            row_d = distances[i]
            row_i = indexes[i]
            row_chunks = [self.get_chunk(chunk_id) for chunk_id in row_i]
            row_r_chunks = [RetrievedChunk(chunk, distance) for chunk, distance in zip(row_chunks, row_d)]
            row_r_chunks = RetrievedChunk.remove_duplicates_and_overlaps(row_r_chunks, self.overlap_threshold,
                                                                         expected_key_db_top_k)
            prelim_r_chunks.append(row_r_chunks)
        return self._multi_retrieve_for_r_chunks(queries, prelim_r_chunks, k,
                                                 show_progress_bar=show_progress_bar)

    def retrieve_multiple(self, queries: List[str], k: int, rewrite: bool = False, show_progress_bar: bool = False,
                          **kwargs) -> List[List[RetrievedMemory]]:
        if self.max_query_length is not None:
            queries = self._truncate_queries(queries, self.max_query_length)
        rk = self.get_retrieval_key_for_text(queries, show_progress_bar=show_progress_bar)
        batch_size, num_rk, emb_size = rk.size(0), rk.size(1), rk.size(2),
        if num_rk != 1:
            raise ValueError('Memory does not support multiple retrieval embeddings')
        n_queries = len(queries)
        if batch_size != n_queries:
            raise SystemError(f'Batch returned by embeddings model is of shape {rk.shape} '
                              f'while the number of queries is {n_queries}')
        rk_np = rk.view(batch_size, emb_size).detach().cpu().numpy()
        expected_key_db_top_k = k
        if self.has_match_prob_model:
            expected_key_db_top_k = round(expected_key_db_top_k * self.reranking_k_factor)
        # The vector_db can return multiple entries with the same ID if chunks are associated
        # with multiple embeddings
        downstream_top_k = expected_key_db_top_k * self.num_storage_embeddings
        # Extra items retrieved, anticipating possible overlaps
        # TODO assumes chunk expansion is to at most 1 chunk on each side
        cc = self.chunk_capacity
        max_extra_passage_len = cc * 2
        num_possible_side_chunks = round((max_extra_passage_len / cc) / (1 - self.overlap_fraction))
        downstream_top_k *= (1 + num_possible_side_chunks)
        distances, indexes = self.vector_db.search(rk_np, k=downstream_top_k)
        # Assumes vector_db returns ordered results
        assert distances.shape[0] == indexes.shape[0] == batch_size
        return self._multi_retrieve(queries, distances, indexes, expected_key_db_top_k, k,
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


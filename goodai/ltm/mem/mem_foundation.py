import abc
import enum
import io
import json
import math
import sys
from dataclasses import dataclass
from typing import List, Union, Set, Optional, Tuple

import faiss
import numpy as np
import torch
from transformers import PreTrainedTokenizer

from goodai.ltm.mem.base import BaseTextMemory, RetrievedMemory, BaseReranker
from goodai.ltm.mem.chunk import Chunk
from goodai.ltm.mem.chunk_queue import PassageInfo, ChunkExpansionOptions
from goodai.ltm.mem.rewrite_model import BaseRewriteModel
from goodai.ltm.mem.simple_vector_db import SimpleVectorDb

_vector_db_type = Union[faiss.Index, SimpleVectorDb]


class VectorDbType(enum.Enum):
    SIMPLE = 0,
    FAISS_FLAT_L2 = 1,


@dataclass
class RetrievedChunk:
    chunk: Chunk
    distance: float
    passage: PassageInfo
    confidence: Optional[float] = None

    def with_confidence(self, confidence: float) -> 'RetrievedChunk':
        return RetrievedChunk(chunk=self.chunk, distance=self.distance,
                              passage=self.passage,
                              confidence=confidence)

    def sort_key(self):
        return self.distance if self.confidence is None else -self.confidence

    @staticmethod
    def has_overlap(included_indexes: Set[int], p_from: int, p_to: int,
                    overlap_threshold: float):
        if p_from not in included_indexes and (p_to - 1) not in included_indexes:
            return False
        if p_to <= p_from:
            return False
        intersection = included_indexes.intersection(range(p_from, p_to))
        len_inter = len(intersection)
        if len_inter <= 0:
            return False
        overlap_fraction = len_inter / (p_to - p_from)
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
            passage = item.passage
            p_from = passage.fromIndex
            p_to = passage.toIndex
            if cls.has_overlap(included_indexes, p_from, p_to, overlap_threshold):
                continue
            result.append(item)
            if len(result) >= max_count:
                break
            included_indexes.update(range(p_from, p_to))
        return result

    @classmethod
    def reduce(cls, items: List['RetrievedChunk'], overlap_threshold: float,
               expected_non_redundant_count: int) -> List['RetrievedChunk']:
        # Does not remove overlapping passages.
        # Returns as many passages as would be needed to get expected_non_redundant_count
        # after removal of overlapping passages.
        id_set = set()
        result = []
        included_indexes: Set[int] = set()
        count = 0
        for item in items:
            chunk = item.chunk
            if chunk is None:
                continue
            chunk_id = chunk.chunk_id
            if chunk_id in id_set:
                continue
            id_set.add(chunk_id)
            passage = item.passage
            p_from = passage.fromIndex
            p_to = passage.toIndex
            result.append(item)
            if not cls.has_overlap(included_indexes, p_from, p_to, overlap_threshold):
                count += 1
                if count >= expected_non_redundant_count:
                    break
            included_indexes.update(range(p_from, p_to))
        return result


class BaseTextMemoryFoundation(BaseTextMemory):
    def __init__(self, vector_db_type: VectorDbType, tokenizer: PreTrainedTokenizer, has_match_prob_model: bool,
                 num_storage_embeddings: int, emb_dim: int,
                 chunk_capacity: int, reranking_k_factor: float,
                 max_query_length: Optional[int],
                 query_rewrite_model: BaseRewriteModel, reranker: BaseReranker,
                 overlap_fraction: float,
                 overlap_threshold: float, chunk_expansion_options: ChunkExpansionOptions,
                 device: torch.device, max_expansion_top_k_factor: int = 200):
        super().__init__()
        if overlap_fraction < 0 or overlap_fraction > 0.5:
            raise ValueError(f'Invalid chunk overlap fraction: {overlap_fraction}')
        if overlap_threshold <= 0 or overlap_threshold > 1.0:
            raise ValueError(f'Invalid redundancy overlap threshold: {overlap_threshold}')
        if reranking_k_factor < 1:
            raise ValueError('reranking_k_factor cannot be less than 1')
        max_st = chunk_expansion_options.maxSideTokens
        if max_st < chunk_expansion_options.minSideTokens:
            raise ValueError('Invalid chunk expansion configuration: The maximum number of extra side tokens '
                             'cannot be less than the minimum number.')
        max_added_chunks = max_st * 2 / chunk_capacity
        max_chunks_overlap = (max_added_chunks + 2) / (1 - overlap_fraction)
        self.expansion_top_k_factor = overlap_threshold + (1 - overlap_threshold) * (1 + max_chunks_overlap)
        if self.expansion_top_k_factor > max_expansion_top_k_factor:
            raise ValueError('Excessive chunk expansion configuration. Increase the chunk capacity or the redundancy '
                             'overlap threshold, or decrease the maximum extra side tokens or the chunk '
                             'overlap fraction.')
        self.chunk_expansion_options = chunk_expansion_options
        self.chunk_capacity = chunk_capacity
        self.overlap_threshold = overlap_threshold
        self.overlap_fraction = overlap_fraction
        self.max_query_length = max_query_length
        self.reranking_k_factor = reranking_k_factor
        self.num_storage_embeddings = num_storage_embeddings
        self.has_match_prob_model = has_match_prob_model
        self.emb_dim = emb_dim
        self.vector_db = self.create_vector_db(vector_db_type, emb_dim)
        self.device = device
        self.chunk_tokenizer = tokenizer
        self.query_rewrite_model = query_rewrite_model
        self.reranker = reranker

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

    @abc.abstractmethod
    def get_complete_passage(self, chunk: Chunk) -> PassageInfo:
        pass

    def dump(self, stream: io.TextIOBase = sys.stdout):
        chunks = self.get_all_chunks()
        stream.write('| Id | Metadata | Importance | Content |\n')
        stream.write('| ----- | -------- | ---------- | ------- |\n')
        for chunk in chunks:
            chunk_text = self.get_chunk_text(chunk)
            ct_js = json.dumps(chunk_text)
            importance_text = 'None' if chunk.importance is None else f'{chunk.importance:.2g}'
            stream.write(f'| {chunk.chunk_id} | {chunk.metadata} | | {importance_text} | {ct_js} |')
            stream.write('\n')
        stream.flush()

    @staticmethod
    def _distance_to_relevance(distance: float, confidence: Optional[float]) -> float:
        if confidence is not None:
            return confidence
        # Assuming embeddings are always unit vectors.
        # Also assuming vector DB (like FAISS) returns squared Euclidean distance.
        # So distance can go from 0 to 4.
        return 1 - distance / 4.0

    def _retrieve_for_processed_r_chunks(self, processed_r_chunks: List[RetrievedChunk], k: int) -> \
            List[RetrievedMemory]:
        if not self.reranker:
            processed_r_chunks = processed_r_chunks[:k]
        has_pm = self.has_match_prob_model
        sequences = [r_chunk.passage.tokenIds for r_chunk in processed_r_chunks]
        retrieved_texts = self.chunk_tokenizer.batch_decode(sequences, skip_special_tokens=True)
        result = []
        for r_chunk, r_text in zip(processed_r_chunks, retrieved_texts):
            confidence = r_chunk.confidence
            chunk = r_chunk.chunk
            metadata = chunk.metadata
            importance = chunk.importance
            distance = r_chunk.distance
            relevance = self._distance_to_relevance(distance, confidence)
            if not has_pm:
                confidence = None
            result.append(RetrievedMemory(passage=r_text.strip(), timestamp=chunk.timestamp,
                                          distance=distance, relevance=relevance, confidence=confidence,
                                          metadata=metadata, importance=importance))
        if self.reranker:
            result = self.reranker.rerank(result, self)
            result = result[:k]
        return result

    def _multi_retrieve_for_r_chunks(self, queries: List[str], prelim_r_chunks: List[List[RetrievedChunk]],
                                     reranking_top_k: int, k: int,
                                     show_progress_bar: bool = False) -> List[List[RetrievedMemory]]:
        # At this point:
        # - No chunks are None
        # - Duplicate chunk IDs have already been removed in prelim_r_chunks
        # - Overlapping chunks are yet to be removed
        # - prelim_r_chunks is ordered by distance
        if self.has_match_prob_model:
            m_sentences = []
            m_indexes = []
            reduced_r_chunks = []
            for i, (query, row_r_chunks) in enumerate(zip(queries, prelim_r_chunks)):
                row_reduced_r_chunks = RetrievedChunk.reduce(row_r_chunks, self.overlap_threshold,
                                                             reranking_top_k)
                reduced_r_chunks.append(row_reduced_r_chunks)
                prelim_chunks = [r_chunk.chunk for r_chunk in row_reduced_r_chunks]
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
            for chunk_probs, row_r_chunks in zip(chunk_probs_list, reduced_r_chunks):
                new_row_r_chunks = [r_chunk.with_confidence(cp) for cp, r_chunk in zip(chunk_probs, row_r_chunks)]
                new_row_r_chunks.sort(key=RetrievedChunk.sort_key)
                processed_r_chunks.append(new_row_r_chunks)
        else:
            processed_r_chunks = prelim_r_chunks
        result = []
        for row_r_chunks in processed_r_chunks:
            # Removal of overlapping passages
            row_r_chunks = RetrievedChunk.remove_duplicates_and_overlaps(row_r_chunks, self.overlap_threshold,
                                                                         reranking_top_k)
            # At this point there are at most reranking_top_k chunks per row
            # TODO custom reranker would go here
            result.append(self._retrieve_for_processed_r_chunks(row_r_chunks, k=k))
        return result

    def _multi_retrieve(self, queries: List[str], distances: np.ndarray, indexes: np.ndarray,
                        reranking_top_k: int, expansion_top_k: int, k: int,
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
            # Remove duplicate and "not found" chunk IDs
            distinct_d_i = []
            id_set = set()
            for distance, chunk_id in zip(row_d, row_i):
                if chunk_id in id_set or chunk_id < 0:
                    continue
                id_set.add(chunk_id)
                distinct_d_i.append((distance, chunk_id,))
            # Without duplicates, we only need at most expansion_top_k chunks at this point
            distinct_d_i = distinct_d_i[:expansion_top_k]
            # Retrieve chunk objects from queue
            row_chunks = [self.get_chunk(chunk_id) for _, chunk_id in distinct_d_i]
            row_passages = [self.get_complete_passage(chunk) for chunk in row_chunks]
            row_r_chunks = [RetrievedChunk(chunk, distance, passage)
                            for chunk, passage, (distance, _) in
                            zip(row_chunks, row_passages, distinct_d_i)]
            prelim_r_chunks.append(row_r_chunks)
        return self._multi_retrieve_for_r_chunks(queries, prelim_r_chunks, reranking_top_k, k,
                                                 show_progress_bar=show_progress_bar)

    def retrieve_multiple(self, queries: List[str], k: int, rewrite: bool = False, show_progress_bar: bool = False,
                          **kwargs) -> List[List[RetrievedMemory]]:
        if k <= 0:
            raise ValueError('k must be greater than zero')
        if rewrite and not self.query_rewrite_model:
            raise ValueError("For query rewriting, a rewriting model must be provided")
        if rewrite and self.query_rewrite_model:
            queries = [self.query_rewrite_model.rewrite_query(q) for q in queries]
        if self.max_query_length is not None:
            queries = self._truncate_queries(queries, self.max_query_length)
        rk = self.get_retrieval_key_for_text(queries, show_progress_bar=show_progress_bar)
        batch_size, num_rk, emb_size = rk.size(0), rk.size(1), rk.size(2),
        if num_rk != 1:
            raise RuntimeError('Memory does not support multiple retrieval embeddings')
        n_queries = len(queries)
        if batch_size != n_queries:
            raise RuntimeError(f'Batch returned by embeddings model is of shape {rk.shape} '
                               f'while the number of queries is {n_queries}')
        rk_np = rk.view(batch_size, emb_size).detach().cpu().numpy()
        reranking_top_k = k
        if self.has_match_prob_model:
            reranking_top_k = round(reranking_top_k * self.reranking_k_factor)

        # Extra items retrieved, anticipating possible overlaps
        expansion_top_k = 1 + math.ceil((reranking_top_k - 1) * self.expansion_top_k_factor)

        # The vector_db can return multiple entries with the same ID if chunks are associated
        # with multiple embeddings
        coverage_top_k = 1 + (expansion_top_k - 1) * self.num_storage_embeddings

        # Call to vector database
        distances, indexes = self.vector_db.search(rk_np, k=coverage_top_k)

        # Assumes vector_db returns ordered results
        assert distances.shape[0] == indexes.shape[0] == batch_size
        return self._multi_retrieve(queries, distances, indexes, reranking_top_k, expansion_top_k, k,
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


import abc
from typing import List, Union, Any, Set, Optional

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from goodai.helpers.collections_helper import remove_duplicates, num_visited_to_get_expected_count, \
    get_non_adjacent
from goodai.helpers.tokenizer_helper import get_sentence_punctuation_ids
from goodai.ltm.mem.base import BaseTextMemory, RetrievedMemory
from goodai.ltm.mem.simple_vector_db import SimpleVectorDb

_vector_db_type = Union[faiss.Index, SimpleVectorDb]


class BaseTextMemoryFoundation(BaseTextMemory):
    def __init__(self, vector_db: _vector_db_type, tokenizer: PreTrainedTokenizer, has_match_prob_model: bool,
                 num_storage_embeddings: int,
                 device: torch.device, adjacent_chunks_ok: bool):
        super().__init__()
        self.num_storage_embeddings = num_storage_embeddings
        self.has_match_prob_model = has_match_prob_model
        self.vector_db = vector_db
        self.adjacent_chunks_ok = adjacent_chunks_ok
        self.device = device
        self.em_tokenizer = tokenizer
        self.punctuation_ids = get_sentence_punctuation_ids(self.em_tokenizer, include_line_break=False)

    def get_tokenizer(self):
        return self.em_tokenizer

    @abc.abstractmethod
    def retrieve_chunk_sequences(self, chunk_ids: List[int]):
        pass

    @abc.abstractmethod
    def retrieve_complete_sequences(self, chunk_ids: List[int], punctuation_ids: Set[int]):
        pass

    @abc.abstractmethod
    def get_retrieval_key_for_text(self, queries: List[str], show_progress_bar: bool = False) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_match_probabilities(self, query: str, passages: List[str]) -> List[float]:
        pass

    @abc.abstractmethod
    def get_metadata(self, chunk_id: int) -> Optional[Any]:
        pass

    def _retrieve(self, query: str, rewrite: bool, flat_distances: np.ndarray, flat_indexes: np.ndarray,
                  expected_key_db_top_k: int, k: int) -> List[RetrievedMemory]:
        # TODO optimize matching batch
        # distances, indexes: (batch_size, downstream_top_k)
        adjacent_chunks_ok = self.adjacent_chunks_ok
        prelim_dist_indexes = list(zip(flat_distances, flat_indexes))
        prelim_dist_indexes = remove_duplicates(prelim_dist_indexes, key_fn=lambda _t: _t[1])
        has_pm = self.has_match_prob_model
        if has_pm:
            nv = num_visited_to_get_expected_count(prelim_dist_indexes, expected_key_db_top_k, adjacent_chunks_ok,
                                                   key_fn=lambda _t: _t[1])
            prelim_dist_indexes = prelim_dist_indexes[:nv * 2]
            prelim_chunk_indexes = [ci for _, ci in prelim_dist_indexes]
            chunk_sequences: List[List[int]] = self.retrieve_chunk_sequences(prelim_chunk_indexes)
            chunk_texts = self.em_tokenizer.batch_decode(chunk_sequences, skip_special_tokens=True)
            chunk_prob = self.get_match_probabilities(query, chunk_texts)
            chunk_score_tuples = list(zip(chunk_prob, prelim_dist_indexes))
            chunk_score_tuples.sort(key=lambda _t: _t[0], reverse=True)
        else:
            pseudo_scores = list(range(len(prelim_dist_indexes), 0, -1))
            chunk_score_tuples = list(zip(pseudo_scores, prelim_dist_indexes))
        if not adjacent_chunks_ok:
            chunk_score_tuples = get_non_adjacent(chunk_score_tuples, key_fn=lambda _t: _t[1][1])
        chunk_score_tuples = chunk_score_tuples[:k]
        final_indexes = [ci for _, (_, ci) in chunk_score_tuples]
        sequences = self.retrieve_complete_sequences(final_indexes, self.punctuation_ids)
        retrieved_texts = self.em_tokenizer.batch_decode(sequences, skip_special_tokens=True)
        result = []
        for cs_tuple, r_text in zip(chunk_score_tuples, retrieved_texts):
            confidence, (distance, chunk_id) = cs_tuple
            metadata = self.get_metadata(chunk_id)
            if not has_pm:
                confidence = None
            result.append(RetrievedMemory(passage=r_text.strip(), distance=distance,
                                          confidence=confidence, metadata=metadata))
        return result

    def retrieve_multiple(self, queries: List[str], k: int = 1, rewrite: bool = False, mm_multiplier: int = 10,
                          show_progress_bar: bool = False) -> List[List[RetrievedMemory]]:
        rk = self.get_retrieval_key_for_text(queries, show_progress_bar=show_progress_bar)
        batch_size, num_rk, emb_size = rk.size(0), rk.size(1), rk.size(2),
        if num_rk != 1:
            raise ValueError('Memory does not support multiple retrieval embeddings')
        rk_np = rk.view(batch_size, emb_size).detach().cpu().numpy()
        adjacent_chunks_ok = self.adjacent_chunks_ok
        expected_key_db_top_k = k
        if self.has_match_prob_model:
            expected_key_db_top_k *= mm_multiplier
        downstream_top_k = expected_key_db_top_k * self.num_storage_embeddings
        if not adjacent_chunks_ok:
            downstream_top_k *= 3
        distances, indexes = self.vector_db.search(rk_np, k=downstream_top_k)
        assert distances.shape[0] == indexes.shape[0] == batch_size
        result = []
        rng = range(batch_size)
        if show_progress_bar:
            rng = tqdm(rng, desc='Retrieval', unit='query')
        for i in rng:
            flat_distances = distances[i]
            flat_indexes = indexes[i]
            single_result = self._retrieve(queries[i], rewrite, flat_distances, flat_indexes, expected_key_db_top_k, k)
            result.append(single_result)
        return result

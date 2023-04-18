import abc
from typing import List, Union, Any, Set, Optional

import faiss
import torch
from transformers import PreTrainedTokenizer

from pyltm.helpers.collections_helper import remove_duplicates, num_visited_to_get_expected_count, \
    get_non_adjacent
from pyltm.helpers.tokenizer_helper import get_sentence_punctuation_ids
from pyltm.memory import BaseTextMemory, RetrievedMemory
from pyltm.memory_models.simple_vector_db import SimpleVectorDb

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
    def get_retrieval_key_for_text(self, queries: List[str]) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_match_probabilities(self, query: str, passages: List[str]) -> List[float]:
        pass

    @abc.abstractmethod
    def get_metadata(self, chunk_id: int) -> Optional[Any]:
        pass

    def retrieve(self, query: str, k: int = 1, mm_multiplier: int = 10) -> List[RetrievedMemory]:
        rk = self.get_retrieval_key_for_text([query])
        batch_size, num_rk = rk.size(0), rk.size(1),
        if num_rk != 1:
            raise ValueError('Memory does not support multiple retrieval embeddings')
        rk_np = rk.view(batch_size * num_rk, -1).detach().cpu().numpy()
        adjacent_chunks_ok = self.adjacent_chunks_ok
        expected_key_db_top_k = k
        if self.has_match_prob_model:
            expected_key_db_top_k *= mm_multiplier
        downstream_top_k = expected_key_db_top_k * self.num_storage_embeddings
        if not adjacent_chunks_ok:
            downstream_top_k *= 3
        distances, indexes = self.vector_db.search(rk_np, k=downstream_top_k)
        prelim_dist_indexes = list(zip(distances.flatten(), indexes.flatten()))
        prelim_dist_indexes = remove_duplicates(prelim_dist_indexes, key_fn=lambda _t: _t[1])
        if self.has_match_prob_model:
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
            result.append(RetrievedMemory(passage=r_text, distance=distance,
                                          confidence=confidence, metadata=metadata))
        return result

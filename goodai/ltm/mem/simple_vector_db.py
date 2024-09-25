from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SimpleVectorDb:
    """
    Simple vector database implementation.
    """
    all_vectors: Optional[np.ndarray] = None
    all_ids: Optional[np.ndarray] = None

    def search(self, vectors: np.ndarray, k: int = 1, max_batch_size=256) -> Tuple[np.ndarray, np.ndarray]:
        dist_list = []
        idx_list = []
        for b0 in range(0, vectors.shape[0], max_batch_size):
            b_vectors = vectors[b0:b0 + max_batch_size]
            b_dist, b_idx = self._search_direct(b_vectors, k=k)
            dist_list.append(b_dist)
            idx_list.append(b_idx)
        return np.concatenate(dist_list), np.concatenate(idx_list),

    def _search_direct(self, vectors: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = vectors.shape[0]
        placeholder_dist = np.ones((batch_size, k)) * 1e+38
        placeholder_ids = -np.ones((batch_size, k), dtype=np.int64)
        if self.all_vectors is None:
            return placeholder_dist, placeholder_ids,
        # vectors: (n, emb_size,)
        # all_vectors: (m, emb_size,)
        diff_sq = (vectors[:, None, :] - self.all_vectors[None, :, :]) ** 2
        # diff_sq: (n, m, emb_size,)
        all_sq_distances = np.sum(diff_sq, axis=2)
        # mean_diff_sq: (n, m,)
        sort_indexes = np.argsort(all_sq_distances, axis=1)
        # sort_indexes: (n, m,)
        select_indexes = sort_indexes[:, :k]
        flat_select_indexes = select_indexes.flatten()
        flat_result_ids = self.all_ids[flat_select_indexes]
        # result_indexes: (n, k,)
        result_size, num_result_ids = select_indexes.shape
        result_range = np.arange(0, result_size)
        rep_range = np.repeat(result_range, num_result_ids)
        result_distances = all_sq_distances[rep_range, flat_select_indexes]
        result_distances = np.reshape(result_distances, select_indexes.shape)
        result_ids = np.reshape(flat_result_ids, select_indexes.shape)
        placeholder_dist[:, :result_distances.shape[1]] = result_distances
        placeholder_ids[:, :result_ids.shape[1]] = result_ids
        return placeholder_dist, placeholder_ids,

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray):
        if vectors.shape[0] != ids.shape[0]:
            raise ValueError('Mismatch in number of vectors and ids provided')
        self.remove_ids(ids)
        assert len(ids.shape) == 1, 'ids must be 1-dimensional'
        if self.all_vectors is None:
            self.all_vectors = vectors
        else:
            self.all_vectors = np.concatenate([self.all_vectors, vectors], axis=0)
        if self.all_ids is None:
            self.all_ids = ids
        else:
            self.all_ids = np.concatenate([self.all_ids, ids], axis=0)

    def reset(self):
        self.all_vectors = None
        self.all_ids = None

    def remove_ids(self, ids: np.ndarray):
        if len(ids.shape) != 1:
            raise ValueError('ids must be 1-dimensional')
        to_remove = np.isin(self.all_ids, ids)
        if np.any(to_remove):
            if np.all(to_remove):
                self.reset()
            else:
                keep = ~to_remove
                self.all_ids = self.all_ids[keep]
                self.all_vectors = self.all_vectors[keep, :]

    def __eq__(self, other):
        if not isinstance(other, SimpleVectorDb):
            return False
        return np.array_equal(self.all_vectors, other.all_vectors) and np.array_equal(self.all_ids, other.all_ids)

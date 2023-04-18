from typing import Optional, Tuple

import numpy as np


class SimpleVectorDb:
    def __init__(self):
        super().__init__()
        self.all_vectors: Optional[np.ndarray] = None
        self.all_ids: Optional[np.ndarray] = None

    def search(self, vectors: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if self.all_vectors is None:
            return np.array([]), np.array([], dtype=np.int64),
        # vectors: (n, emb_size,)
        # all_vectors: (m, emb_size,)
        diff_sq = (vectors[:, None, :] - self.all_vectors[None, :, :]) ** 2
        # diff_sq: (n, m, emb_size,)
        mean_diff_sq = np.mean(diff_sq, axis=2)
        # mean_diff_sq: (n, m,)
        sort_indexes = np.argsort(mean_diff_sq, axis=1)
        select_indexes = sort_indexes[:, :k]
        flat_select_indexes = select_indexes.flatten()
        flat_result_ids = self.all_ids[flat_select_indexes]
        # result_indexes: (n, k,)
        result_size, num_result_ids = select_indexes.shape
        result_range = np.arange(0, result_size)
        rep_range = np.repeat(result_range, num_result_ids)
        result_distances = mean_diff_sq[rep_range, flat_select_indexes]
        result_distances = np.reshape(result_distances, select_indexes.shape)
        result_ids = np.reshape(flat_result_ids, select_indexes.shape)
        return result_distances, result_ids,

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

from typing import List, Set


class ChunkMixin:
    def __init__(self):
        super().__init__()

    @staticmethod
    def _from_last_punctuation(token_ids: List[int], punctuation_ids: Set[int]):
        for i in range(len(token_ids) - 1, -1, -1):
            token_id = token_ids[i]
            if token_id in punctuation_ids:
                return token_ids[i + 1:]
        return token_ids

    @staticmethod
    def _to_first_punctuation(token_ids: List[int], punctuation_ids: Set[int]):
        for i, token_id in enumerate(token_ids):
            if token_id in punctuation_ids:
                return token_ids[:i + 1]
        return token_ids


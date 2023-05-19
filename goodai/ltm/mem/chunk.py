import time
from typing import Any, Optional


class Chunk:
    def __init__(self, chunk_id: int, capacity: int, from_token_seq_id: int, metadata: Optional[dict]):
        self.metadata = metadata
        self.chunk_id = chunk_id
        self.capacity = capacity
        self.from_token_seq_id = from_token_seq_id
        self.to_token_seq_id = from_token_seq_id
        self.indexed: bool = False
        self.timestamp: float = time.time()

    def __len__(self):
        return self.to_token_seq_id - self.from_token_seq_id

    def get_room(self):
        return self.capacity - len(self)

    def is_at_capacity(self) -> bool:
        return len(self) >= self.capacity

    def set_to_token_seq_id(self, to_token_seq_id: int):
        if to_token_seq_id - self.from_token_seq_id > self.capacity:
            raise ValueError(f'capacity={self.capacity} would be exceeded')
        self.to_token_seq_id = to_token_seq_id

    def extend_by(self, num_tokens: int):
        if len(self) + num_tokens > self.capacity:
            raise ValueError(f'capacity={self.capacity} would be exceeded')
        self.to_token_seq_id += num_tokens

    def is_indexed(self) -> bool:
        return self.indexed

    def set_indexed(self, mode: bool):
        self.indexed = mode

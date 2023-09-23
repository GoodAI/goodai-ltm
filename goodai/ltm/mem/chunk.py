from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

TextKeyType = int


@dataclass
class Chunk:
    chunk_id: int
    capacity: int
    from_token_seq_id: int
    metadata: Optional[Dict[Any, Any]]
    importance: Optional[float]
    timestamp: float
    to_token_seq_id: int = None
    indexed_length: int = field(default=-1)
    associated_keys: List[TextKeyType] = field(default_factory=list)

    def __post_init__(self):
        if not self.to_token_seq_id:
            self.to_token_seq_id = self.from_token_seq_id

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

    def shift(self, offset: int):
        self.from_token_seq_id += offset
        self.to_token_seq_id += offset

    def is_indexed(self) -> bool:
        return self.indexed_length == len(self)

    def update_indexed_state(self):
        self.indexed_length = len(self)

    def add_key(self, text_key: TextKeyType):
        self.associated_keys.append(text_key)

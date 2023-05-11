import math
from abc import abstractmethod, ABC
from typing import List, Tuple, Set, Dict, Optional, Any

from goodai.ltm.mem.chunk import Chunk
from goodai.ltm.mem.chunk_mixin import ChunkMixin


class BaseChunkQueue(ABC):
    """
    Abstract base class for text memory chunk queues.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_capacity(self) -> int:
        pass

    @abstractmethod
    def add_sequence(self, token_ids: List[int], metadata: Optional[Any]):
        pass

    @abstractmethod
    def check_overflow(self) -> List[Chunk]:
        pass

    @abstractmethod
    def get_chunks_for_indexing(self) -> Tuple[List[Chunk], List[List[int]]]:
        pass

    @abstractmethod
    def get_latest_token_ids(self, max_num_tokens: Optional[int]):
        pass

    @abstractmethod
    def retrieve_chunk_sequences(self, chunk_ids: List[int]):
        pass

    @abstractmethod
    def retrieve_complete_sequences(self, chunk_ids: List[int], punctuation_ids: Set[int]):
        pass

    @abstractmethod
    def flush(self):
        pass

    @abstractmethod
    def get_chunk_sequences(self) -> List[List[int]]:
        pass

    @abstractmethod
    def get_chunk(self, chunk_id: int) -> Chunk:
        pass

    @abstractmethod
    def get_all_chunks(self) -> List[Chunk]:
        pass

    @abstractmethod
    def get_chunk_token_ids(self, chunk: Chunk):
        pass


class ChunkQueue(BaseChunkQueue, ChunkMixin):
    def __init__(self, queue_capacity: int, chunk_capacity: int, chunk_index_at_overlap: int,
                 first_token_seq_id: int = 0):
        super().__init__()
        if queue_capacity <= 2:
            raise ValueError('Queue capacity cannot be 2 or less')
        if chunk_capacity < 1:
            raise ValueError('Chunk capacity cannot be zero or less')
        if chunk_index_at_overlap < math.ceil(chunk_capacity / 2):
            raise ValueError('Chunk overlap cannot be more than 50%')
        self.capacity = queue_capacity
        self.chunk_capacity = chunk_capacity
        self.chunks: List[Chunk] = []
        self.min_tokens_for_indexing = chunk_capacity // 2
        self.chunk_index_at_overlap = chunk_index_at_overlap
        self.current_chunk_id = 0
        self.first_token_seq_id = first_token_seq_id
        self.token_ids = []
        self.chunk_map: Dict[int, Chunk] = dict()

    def _pop_chunk(self):
        chunk = self.chunks.pop(0)
        if chunk is not None:
            self.chunk_map.pop(chunk.chunk_id)
            if len(self.chunks) == 0:
                assert len(self.chunk_map) == 0
                self.token_ids = []
                self.first_token_seq_id = 0
            else:
                new_first_chunk = self.chunks[0]
                new_first_token_seq_id = new_first_chunk.from_token_seq_id
                num_removed = new_first_token_seq_id - self.first_token_seq_id
                self.token_ids = self.token_ids[num_removed:]
                self.first_token_seq_id = new_first_token_seq_id
        return chunk

    def get_all_chunks(self) -> List[Chunk]:
        return list(self.chunks)

    def check_overflow(self) -> List[Chunk]:
        removed_chunks = []
        while len(self.chunks) > self.capacity:
            removed_chunk = self._pop_chunk()
            if removed_chunk is not None:
                removed_chunks.append(removed_chunk)
        return removed_chunks

    def add_chunk(self, metadata: Optional[Any]) -> Chunk:
        chunk_id = self.current_chunk_id
        last_chunk = self.chunks[-1] if len(self.chunks) >= 1 else None
        if last_chunk is None:
            from_token_seq_id = self.first_token_seq_id
        else:
            from_token_seq_id = last_chunk.from_token_seq_id + self.chunk_index_at_overlap
        chunk = Chunk(chunk_id, self.chunk_capacity, from_token_seq_id, metadata)
        self.current_chunk_id = chunk_id + 1
        self.chunks.append(chunk)
        self.chunk_map[chunk.chunk_id] = chunk
        self.check_overflow()
        return chunk

    def get_chunk(self, chunk_id: int) -> Chunk:
        return self.chunk_map.get(chunk_id)

    def extend_chunk(self, chunk: Chunk, token_ids: List[int], new_tokens: bool):
        if new_tokens:
            self.token_ids.extend(token_ids)
        chunk.extend_by(len(token_ids))

    def get_first_token_sequence_id(self) -> int:
        return self.first_token_seq_id

    def get_next_token_sequence_id(self) -> int:
        return self.first_token_seq_id + len(self.token_ids)

    def ensure_chunks_exist(self, metadata: Optional[Any]):
        while len(self.chunks) < 2:
            self.add_chunk(metadata)

    def add_sequence(self, new_token_ids: List[int], metadata: Optional[Any]) -> List[Chunk]:
        self.ensure_chunks_exist(metadata)
        self.token_ids += new_token_ids
        next_token_seq_id = len(self.token_ids) + self.first_token_seq_id
        c_idx = len(self.chunks) - 2
        while True:
            while c_idx >= len(self.chunks):
                self.add_chunk(metadata)
            chunk = self.chunks[c_idx]
            if chunk.to_token_seq_id < next_token_seq_id:
                chunk.extend_by(min(next_token_seq_id - chunk.to_token_seq_id, chunk.get_room()))
            elif len(chunk) <= self.chunk_index_at_overlap:
                break
        return self.check_overflow()

    def get_chunks_for_indexing(self) -> Tuple[List[Chunk], List[List[int]]]:
        token_id_matrix = []
        picked_buckets = []
        for i, chunk in enumerate(self.chunks):
            if not chunk.is_indexed():
                token_ids = self.get_chunk_token_ids(chunk)
                if len(token_ids) >= self.min_tokens_for_indexing or (i == 0 and len(token_ids) > 0):
                    picked_buckets.append(chunk)
                    token_id_matrix.append(token_ids)
        return picked_buckets, token_id_matrix,

    def get_chunk_token_ids(self, chunk: Chunk):
        first_index = self.first_token_seq_id
        from_index = chunk.from_token_seq_id - first_index
        to_index = chunk.to_token_seq_id - first_index
        return self.token_ids[from_index:to_index]

    def _get_prev_token_ids(self, chunk_id: int) -> Optional[List[int]]:
        chunk = self.chunks[chunk_id]
        first_index = self.first_token_seq_id
        to_index = chunk.from_token_seq_id - first_index
        from_index = to_index - self.chunk_capacity
        from_index = max(0, from_index)
        return self.token_ids[from_index:to_index]

    def _get_next_token_ids(self, chunk_id: int) -> Optional[List[int]]:
        chunk = self.chunks[chunk_id]
        first_index = self.first_token_seq_id
        from_index = chunk.to_token_seq_id - first_index
        to_index = from_index + self.chunk_capacity
        return self.token_ids[from_index:to_index]

    def retrieve_complete_sequences(self, chunk_ids: List[int], punctuation_ids: Set[int]):
        sequences = []
        for chunk_id in chunk_ids:
            chunk = self.chunk_map.get(chunk_id)
            if chunk is not None:
                sequence = self.get_chunk_token_ids(chunk)
                prev_sequence = self._get_prev_token_ids(chunk_id)
                if prev_sequence is not None:
                    sequence = self._from_last_punctuation(prev_sequence, punctuation_ids) + sequence
                next_sequence = self._get_next_token_ids(chunk_id)
                if next_sequence is not None:
                    sequence = sequence + self._to_first_punctuation(next_sequence, punctuation_ids)
                sequences.append(sequence)
        return sequences

    def get_queue_size(self):
        return len(self.chunks)

    def get_chunk_sequences(self) -> List[List[int]]:
        return [self.get_chunk_token_ids(bucket) for bucket in self.chunks]

    def get_capacity(self) -> int:
        return self.capacity

    def get_latest_token_ids(self, max_num_tokens: Optional[int]):
        total = 0
        result = []
        len_queue = len(self.chunks)
        for i in range(len_queue - 1, -1, -2):
            chunk = self.chunks[i]
            token_ids = self.get_chunk_token_ids(chunk)
            new_total = total + len(token_ids)
            if max_num_tokens is not None and new_total >= max_num_tokens:
                remain_count = max_num_tokens - total
                if remain_count > 0:
                    result[:0] = token_ids[-remain_count:]
                break
            else:
                result[:0] = token_ids
            total = new_total
        return result

    def retrieve_chunk_sequences(self, chunk_ids: List[int]):
        sequences = []
        for chunk_id in chunk_ids:
            chunk = self.chunk_map.get(chunk_id)
            if chunk is not None:
                sequence = self.get_chunk_token_ids(chunk)
                sequences.append(sequence)
        return sequences

    def flush(self):
        self.chunks = []
        self.token_ids = []
        self.first_token_seq_id = 0
        self.current_chunk_id = 0
        self.chunk_map.clear()

from abc import abstractmethod, ABC
from typing import List, Tuple, Set, Dict, Optional, Any

from goodai_ltm.memory_models.chunk import Chunk
from goodai_ltm.memory_models.chunk_mixin import ChunkMixin


class BaseChunkQueue(ABC):
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


class ChunkQueue(BaseChunkQueue, ChunkMixin):
    def __init__(self, queue_capacity: int, chunk_capacity: int, first_token_seq_id: int = 0):
        super().__init__()
        assert queue_capacity > 2, 'capacity cannot be 2 or less.'
        self.chunks: List[Chunk] = []
        self.capacity = queue_capacity
        self.chunk_capacity = chunk_capacity
        self.half_chunk_capacity = chunk_capacity // 2
        self.current_chunk_index = 0
        self.first_token_seq_id = first_token_seq_id
        self.token_ids = []
        self.chunk_map: Dict[int, Chunk] = dict()

    def _pop_chunk(self):
        chunk = self.chunks.pop(0)
        if chunk is not None:
            self.chunk_map.pop(chunk.index)
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

    def check_overflow(self) -> List[Chunk]:
        removed_chunks = []
        while len(self.chunks) > self.capacity:
            removed_chunk = self._pop_chunk()
            if removed_chunk is not None:
                removed_chunks.append(removed_chunk)
        return removed_chunks

    def ensure_chunks_exist(self, metadata: Optional[Any]):
        while len(self.chunks) < 2:
            self.add_chunk(metadata)

    def add_chunk(self, metadata: Optional[Any]) -> Chunk:
        b_index = self.current_chunk_index
        last_chunk = self.chunks[-1] if len(self.chunks) >= 1 else None
        prev_last_chunk = self.chunks[-2] if len(self.chunks) >= 2 else None
        first_token_seq_id: int
        if prev_last_chunk is not None:
            first_token_seq_id = prev_last_chunk.from_token_seq_id + self.chunk_capacity
        elif last_chunk is not None:
            first_token_seq_id = last_chunk.from_token_seq_id + self.half_chunk_capacity
        else:
            first_token_seq_id = self.first_token_seq_id
        chunk = Chunk(b_index, self.chunk_capacity, first_token_seq_id, metadata)
        self.current_chunk_index = b_index + 1
        self.chunks.append(chunk)
        self.chunk_map[chunk.index] = chunk
        self.check_overflow()
        return chunk

    def get_chunk(self, chunk_id: int) -> Chunk:
        return self.chunk_map[chunk_id]

    def extend_chunk(self, chunk: Chunk, token_ids: List[int], new_tokens: bool):
        if new_tokens:
            self.token_ids.extend(token_ids)
        chunk.extend_by(len(token_ids))

    def get_first_token_sequence_id(self) -> int:
        return self.first_token_seq_id

    def get_next_token_sequence_id(self) -> int:
        return self.first_token_seq_id + len(self.token_ids)

    def add_sequence(self, token_ids: List[int], metadata: Optional[Any]) -> List[Chunk]:
        self.ensure_chunks_exist(metadata)
        remain_token_ids = token_ids
        while len(remain_token_ids) > 0:
            first_bucket = self.chunks[0]
            if len(first_bucket) < self.half_chunk_capacity:
                initial_room = self.half_chunk_capacity - len(first_bucket)
                leading_token_ids = remain_token_ids[:initial_room]
                self.extend_chunk(first_bucket, leading_token_ids, True)
                remain_token_ids = remain_token_ids[initial_room:]
                continue
            chunk1 = self.chunks[-1]
            chunk2 = self.chunks[-2]
            room2 = chunk2.get_room()
            if room2 <= 0:
                self.add_chunk(metadata)
                chunk1 = self.chunks[-1]
                chunk2 = self.chunks[-2]
                room1 = chunk1.get_room()
                room2 = chunk2.get_room()
            else:
                room1 = chunk1.get_room()
                if len(chunk1) == 0:
                    chunk1.metadata = metadata
            assert room1 > 0 and room2 > 0, f'room1={room1}, room2={room2}, num_buckets={len(self.chunks)}'
            min_room = min(room1, room2)
            if len(remain_token_ids) <= min_room:
                self.extend_chunk(chunk1, remain_token_ids, True)
                self.extend_chunk(chunk2, remain_token_ids, False)
                remain_token_ids = []
            else:
                leading_token_ids = remain_token_ids[:min_room]
                self.extend_chunk(chunk1, leading_token_ids, True)
                self.extend_chunk(chunk2, leading_token_ids, False)
                remain_token_ids = remain_token_ids[min_room:]
        return self.check_overflow()

    def get_chunk_token_ids(self, chunk: Chunk):
        first_index = self.first_token_seq_id
        from_index = chunk.from_token_seq_id - first_index
        to_index = chunk.to_token_seq_id - first_index
        return self.token_ids[from_index:to_index]

    def get_chunks_for_indexing(self) -> Tuple[List[Chunk], List[List[int]]]:
        min_tokens_for_indexing = self.half_chunk_capacity
        token_id_matrix = []
        picked_buckets = []
        for bucket in self.chunks:
            if not bucket.is_indexed():
                token_ids = self.get_chunk_token_ids(bucket)
                if len(token_ids) >= min_tokens_for_indexing:
                    picked_buckets.append(bucket)
                    token_id_matrix.append(token_ids)
        return picked_buckets, token_id_matrix,

    def _get_prev_token_ids(self, chunk_id: int) -> Optional[List[int]]:
        prev_chunk = self.chunk_map.get(chunk_id - 2)
        if prev_chunk is not None:
            return self.get_chunk_token_ids(prev_chunk)
        prev_chunk = self.chunk_map.get(chunk_id - 1)
        if prev_chunk is not None:
            pc_token_ids = self.get_chunk_token_ids(prev_chunk)
            return pc_token_ids[:self.half_chunk_capacity]
        return None

    def _get_next_token_ids(self, chunk_id: int) -> Optional[List[int]]:
        next_chunk = self.chunk_map.get(chunk_id + 2)
        if next_chunk is not None:
            return self.get_chunk_token_ids(next_chunk)
        next_chunk = self.chunk_map.get(chunk_id + 1)
        if next_chunk is not None:
            nc_token_ids = self.get_chunk_token_ids(next_chunk)
            return nc_token_ids[self.half_chunk_capacity:]
        return None

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
        self.current_chunk_index = 0
        self.chunk_map.clear()

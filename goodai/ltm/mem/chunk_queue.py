import bisect
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Set

from transformers import PreTrainedTokenizer

from goodai.ltm.mem.chunk import Chunk, TextKeyType
from goodai.ltm.mem.config import ChunkExpansionConfig


@dataclass
class PassageInfo:
    fromIndex: int
    toIndex: int
    tokenIds: List[int]


@dataclass
class ChunkExpansionOptions:
    minSideTokens: int
    maxSideTokens: int
    leftStopAfterTokenIds: List[List[int]]
    rightStopAtTokenIds: List[List[int]]

    @classmethod
    def default(cls, max_side_tokens: int, punctuation_ids: Set[int]):
        token_ids = [[tid] for tid in punctuation_ids]
        return cls(minSideTokens=0, maxSideTokens=max_side_tokens,
                   leftStopAfterTokenIds=token_ids,
                   rightStopAtTokenIds=token_ids)

    @classmethod
    def from_config(cls, tokenizer: PreTrainedTokenizer, config: ChunkExpansionConfig):
        celt = config.limit_type
        limit_token_ids = celt.get_token_ids(tokenizer)
        return cls(minSideTokens=config.min_extra_side_tokens,
                   maxSideTokens=config.max_extra_side_tokens,
                   leftStopAfterTokenIds=limit_token_ids,
                   rightStopAtTokenIds=limit_token_ids)


class ChunkQueue:
    def __init__(self, queue_capacity: int, chunk_capacity: int, chunk_index_at_overlap: int,
                 first_token_seq_id: int = 0):
        super().__init__()
        if queue_capacity <= 2:
            raise ValueError('Queue capacity cannot be 2 or less')
        if chunk_capacity < 1:
            raise ValueError('Chunk capacity cannot be zero or less')
        if chunk_index_at_overlap < chunk_capacity // 2:
            raise ValueError('Chunk overlap cannot be more than 50%')
        self.capacity = queue_capacity
        self.chunk_capacity = chunk_capacity
        self.chunks: List[Chunk] = []
        self.chunk_index_at_overlap = chunk_index_at_overlap
        self.current_chunk_id = 0
        self.current_text_key = 0
        self.first_token_seq_id = first_token_seq_id
        self.token_ids: List[int] = []
        self.separator_seq_ids: List[int] = []
        self.chunk_map: Dict[int, Chunk] = dict()
        self.sequence_map: Dict[TextKeyType, Tuple[int, int]] = dict()

    def _pop_chunk(self):
        chunk = self.chunks.pop(0)
        if chunk is not None:
            self.chunk_map.pop(chunk.chunk_id)
            if len(self.chunks) == 0:
                assert len(self.chunk_map) == 0
                self.token_ids = []
                self.separator_seq_ids = []
                self.sequence_map.clear()
                self.first_token_seq_id = 0
            else:
                new_first_chunk = self.chunks[0]
                new_first_token_seq_id = new_first_chunk.from_token_seq_id
                num_removed = new_first_token_seq_id - self.first_token_seq_id
                self.token_ids = self.token_ids[num_removed:]
                self.first_token_seq_id = new_first_token_seq_id
                sii = bisect.bisect_left(self.separator_seq_ids, new_first_token_seq_id)
                self.separator_seq_ids = self.separator_seq_ids[sii:]
                for text_key in chunk.associated_keys:
                    del self.sequence_map[text_key]
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

    def add_chunk(self, metadata: Optional[Any], importance: Optional[float], timestamp: float,
                  starts_section: bool = False) -> Chunk:
        chunk_id = self.current_chunk_id
        last_chunk = self.chunks[-1] if len(self.chunks) >= 1 else None
        if last_chunk is None:
            from_token_seq_id = self.first_token_seq_id
        else:
            offset = self.chunk_capacity if starts_section else self.chunk_index_at_overlap
            from_token_seq_id = last_chunk.from_token_seq_id + offset
        chunk = Chunk(chunk_id, self.chunk_capacity, from_token_seq_id, metadata, importance, timestamp)
        self.current_chunk_id = chunk_id + 1
        self.chunks.append(chunk)
        self.chunk_map[chunk.chunk_id] = chunk
        return chunk

    def add_separator(self, pad_token_id: int, timestamp: Optional[float] = None):
        last_chunk = self.chunks[-1] if len(self.chunks) >= 1 else None
        if last_chunk is not None:
            room = last_chunk.get_room()
            pad_seq = [pad_token_id] * room
            self.add_sequence(pad_seq, metadata=None, importance=None, _no_new_chunks=True)
        # separator_seq_ids always assumed to be ordered
        self.separator_seq_ids.append(self.first_token_seq_id + len(self.token_ids))
        if timestamp is None:
            timestamp = time.time()
        self.add_chunk(metadata=None, importance=None, starts_section=True, timestamp=timestamp)

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

    def _new_text_key(self) -> TextKeyType:
        self.current_text_key += 1
        return self.current_text_key

    def add_sequence(self, new_token_ids: List[int], metadata: Optional[Any],
                     importance: Optional[float] = None, timestamp: Optional[float] = None,
                     _no_new_chunks: bool = False) -> List[Chunk]:
        """
        Adds tokens to the chunk queue.
        :param new_token_ids: The sequence of token IDs to add
        :param metadata: A metadata object
        :param importance: An optional importance value for the tokens
        :param timestamp: The timestamp of the stored tokens
        :param _no_new_chunks: If true, attempt should be made to add all tokens without adding new chunks
        :return: Any chunks removed due to overflow.
        """
        if timestamp is None:
            timestamp = time.time()
        text_key = self._new_text_key()
        prev_token_seq_id = len(self.token_ids)
        self.token_ids.extend(new_token_ids)
        next_token_seq_id = len(self.token_ids) + self.first_token_seq_id
        start_num_chunks = len(self.chunks)
        first_c_idx = max(0, start_num_chunks - 2)
        key_registered = False
        for c_idx in range(first_c_idx, start_num_chunks + len(new_token_ids) + 1):
            if _no_new_chunks:
                if c_idx >= len(self.chunks):
                    raise RuntimeError('No new chunks allowed, but at least one more is needed to complete operation')
            else:
                while c_idx >= len(self.chunks):
                    self.add_chunk(metadata, importance, timestamp=timestamp)
            chunk = self.chunks[c_idx]
            if chunk.to_token_seq_id < next_token_seq_id:
                room = chunk.get_room()
                if room > 0:
                    chunk.add_key(text_key)
                    key_registered = True
                    chunk.extend_by(min(next_token_seq_id - chunk.to_token_seq_id, room))
                    much_room = room > self.chunk_capacity // 2
                    if metadata and (chunk.metadata is None or much_room):
                        chunk.metadata = metadata
                    if importance is not None and (chunk.importance is None or much_room):
                        chunk.importance = importance
                    if room == self.chunk_capacity:
                        chunk.timestamp = timestamp
            if (_no_new_chunks or len(chunk) <= self.chunk_index_at_overlap) and \
                    chunk.to_token_seq_id >= next_token_seq_id:
                break
        if key_registered:
            self.sequence_map[text_key] = (prev_token_seq_id, next_token_seq_id,)
        return self.check_overflow()

    def get_chunks_for_indexing(self) -> Tuple[List[Chunk], List[List[int]]]:
        token_id_matrix = []
        picked_buckets = []
        for i, chunk in enumerate(self.chunks):
            if not chunk.is_indexed():
                token_ids = self.get_chunk_token_ids(chunk)
                if len(token_ids) > 0:
                    picked_buckets.append(chunk)
                    token_id_matrix.append(token_ids)
        return picked_buckets, token_id_matrix,

    def get_chunk_token_ids(self, chunk: Chunk):
        first_index = self.first_token_seq_id
        from_index = chunk.from_token_seq_id - first_index
        to_index = chunk.to_token_seq_id - first_index
        return self.token_ids[from_index:to_index]

    def _get_section_bounds(self, from_index: int, to_index: int) -> Tuple[int, int]:
        ssi = bisect.bisect_right(self.separator_seq_ids, from_index)
        ss_seq_id = self.separator_seq_ids[ssi - 1] if ssi >= 1 else self.first_token_seq_id
        esi = bisect.bisect_left(self.separator_seq_ids, to_index)
        es_seq_id = self.separator_seq_ids[esi] if esi < len(self.separator_seq_ids) \
            else self.first_token_seq_id + len(self.token_ids)
        return ss_seq_id, es_seq_id,

    def _get_prev_token_ids(self, chunk: Chunk, section_from_seq_id: int) -> Optional[List[int]]:
        to_index = chunk.from_token_seq_id - self.first_token_seq_id
        from_index = to_index - self.chunk_capacity
        from_index = max(section_from_seq_id - self.first_token_seq_id, from_index)
        return self.token_ids[from_index:to_index]

    def _get_next_token_ids(self, chunk: Chunk, section_to_seq_id: int) -> Optional[List[int]]:
        from_index = chunk.to_token_seq_id - self.first_token_seq_id
        to_index = min(section_to_seq_id - self.first_token_seq_id, from_index + self.chunk_capacity)
        return self.token_ids[from_index:to_index]

    def retrieve_complete_sequences(self, chunk_ids: List[int], options: ChunkExpansionOptions) -> List[List[int]]:
        sequences = []
        for chunk_id in chunk_ids:
            chunk = self.chunk_map.get(chunk_id)
            if chunk is not None:
                passage = self.get_complete_passage(chunk, options)
                sequence = passage.tokenIds
            else:
                sequence = []
            sequences.append(sequence)
        return sequences

    def get_complete_passage(self, chunk: Chunk, options: ChunkExpansionOptions) -> PassageInfo:
        s_from, s_to = self._get_section_bounds(chunk.from_token_seq_id, chunk.to_token_seq_id)
        min_st = options.minSideTokens
        min_p_from, min_p_to = chunk.from_token_seq_id - min_st, chunk.to_token_seq_id + min_st,
        min_p_from, min_p_to = max(min_p_from, s_from), min(min_p_to, s_to)
        max_st = options.maxSideTokens
        max_p_from, max_p_to = chunk.from_token_seq_id - max_st, chunk.to_token_seq_id + max_st,
        max_p_from, max_p_to = max(max_p_from, s_from), min(max_p_to, s_to)
        central_seq = self.get_subsequence(min_p_from, min_p_to)
        prev_token_ids = self.get_subsequence(max_p_from, min_p_from)
        prev_token_ids = self._from_last_match(prev_token_ids, options.leftStopAfterTokenIds)
        next_token_ids = self.get_subsequence(min_p_to, max_p_to)
        next_token_ids = self._to_first_match(next_token_ids, options.rightStopAtTokenIds)
        p_from = min_p_from - len(prev_token_ids)
        p_to = min_p_to + len(next_token_ids)
        expanded_seq = prev_token_ids + central_seq + next_token_ids
        return PassageInfo(p_from, p_to, expanded_seq)

    def get_subsequence(self, from_seq_id: int, to_seq_id: int):
        first = self.first_token_seq_id
        from_idx = from_seq_id - first
        to_idx = to_seq_id - first
        return self.token_ids[from_idx:to_idx]

    @staticmethod
    def _from_last_match(token_ids: List[int], sub_seqs: List[List[int]]):
        nt = len(token_ids)
        match_indexes = []
        for sub_seq in sub_seqs:
            ls = len(sub_seq)
            if ls > 0:
                for i in range(nt - ls, -1, -1):
                    if token_ids[i:i + ls] == sub_seq:
                        match_indexes.append((i, ls,))
                        break
        if len(match_indexes) == 0:
            return token_ids
        max_idx, max_len = max(match_indexes, key=lambda _t: _t[0])
        return token_ids[max_idx + max_len:]

    @staticmethod
    def _to_first_match(token_ids: List[int], sub_seqs: List[List[int]]):
        nt = len(token_ids)
        match_indexes = []
        for sub_seq in sub_seqs:
            ls = len(sub_seq)
            if ls > 0:
                for i in range(0, nt - ls):
                    if token_ids[i:i + ls] == sub_seq:
                        match_indexes.append((i, ls,))
                        break
        if len(match_indexes) == 0:
            return token_ids
        min_idx, min_len = min(match_indexes, key=lambda _t: _t[0])
        return token_ids[:min_idx + min_len]

    def get_queue_size(self):
        return len(self.chunks)

    def get_chunk_sequences(self) -> List[List[int]]:
        return [self.get_chunk_token_ids(chunk) for chunk in self.chunks]

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

    def retrieve_chunk_sequences_given_chunks(self, chunks: List[Chunk]):
        sequences = []
        for chunk in chunks:
            sequence = self.get_chunk_token_ids(chunk)
            sequences.append(sequence)
        return sequences

    def flush(self):
        self.chunks = []
        self.token_ids = []
        self.first_token_seq_id = 0
        self.current_chunk_id = 0
        self.chunk_map.clear()

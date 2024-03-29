import bisect
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from typing import Tuple, Any, Set

import numpy as np
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


@dataclass
class ChunkQueue:
    capacity: int
    chunk_capacity: int
    chunk_index_at_overlap: int
    first_token_seq_id: int = 0
    chunks: List[Chunk] = field(default_factory=list)
    current_chunk_id: int = 0
    current_text_key: int = 0
    token_ids: List[int] = field(default_factory=list)
    separator_seq_ids: List[int] = field(default_factory=list)
    sequence_map: Dict[TextKeyType, Tuple[int, int]] = field(default_factory=dict)
    chunk_map: Dict[int, Chunk] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if self.capacity <= 2:
            raise ValueError('Queue capacity cannot be 2 or less')
        if self.chunk_capacity < 1:
            raise ValueError('Chunk capacity cannot be zero or less')
        if self.chunk_index_at_overlap < self.chunk_capacity // 2:
            raise ValueError('Chunk overlap cannot be more than 50%')
        self.chunk_map = {c.chunk_id: c for c in self.chunks}

    def _removed_chunk_cleanup(self, chunk: Chunk):
        self.chunk_map.pop(chunk.chunk_id)
        if len(self.chunks) == 0:
            assert len(self.chunk_map) == 0
            self.flush()
        else:
            new_first_chunk = self.chunks[0]
            new_first_token_seq_id = new_first_chunk.from_token_seq_id
            num_removed = new_first_token_seq_id - self.first_token_seq_id
            if num_removed != 0:
                self.token_ids = self.token_ids[num_removed:]
                self.first_token_seq_id = new_first_token_seq_id
                sii = bisect.bisect_left(self.separator_seq_ids, new_first_token_seq_id)
                self.separator_seq_ids = self.separator_seq_ids[sii:]
                for text_key in chunk.associated_keys:
                    text_bounds = self.sequence_map.get(text_key)
                    if text_bounds is not None:
                        _, text_to = text_bounds
                        if text_to <= new_first_token_seq_id:
                            del self.sequence_map[text_key]

    def _pop_chunk(self):
        chunk = self.chunks.pop(0)
        if chunk is not None:
            self._removed_chunk_cleanup(chunk)
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
                  starts_section: bool = False, section_seq_id: Optional[int] = None) -> Chunk:
        chunk_id = self.current_chunk_id
        last_chunk = self.chunks[-1] if len(self.chunks) >= 1 else None
        if last_chunk is None:
            from_token_seq_id = self.first_token_seq_id
        elif starts_section:
            assert section_seq_id is not None
            from_token_seq_id = section_seq_id
        else:
            from_token_seq_id = last_chunk.from_token_seq_id + self.chunk_index_at_overlap
        chunk = Chunk(chunk_id, self.chunk_capacity, from_token_seq_id, metadata, importance, timestamp)
        self.current_chunk_id = chunk_id + 1
        self.chunks.append(chunk)
        self.chunk_map[chunk.chunk_id] = chunk
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

    def _new_text_key(self) -> TextKeyType:
        self.current_text_key += 1
        return self.current_text_key

    def _pad_last_chunk(self, pad_token_id: int):
        last_chunk = self.chunks[-1] if len(self.chunks) >= 1 else None
        if last_chunk is not None:
            room = last_chunk.get_room()
            pad_seq = [pad_token_id] * room
            self.add_sequence(pad_seq, metadata=None, importance=None, _no_new_chunks=True)
            return room
        else:
            return 0

    def _update_sequence_map(self, text_key: TextKeyType, new_to_seq_id: int, shift_offset: int):
        # TODO not super efficient
        old_bounds = self.sequence_map.get(text_key)
        if old_bounds is None:
            return
        from_seq_id, old_to_seq_id = old_bounds

        def _shift(b_from_id: int, b_to_id: int):
            if b_from_id >= old_to_seq_id:
                return b_from_id + shift_offset, b_to_id + shift_offset,
            else:
                return b_from_id, b_to_id,

        self.sequence_map = {k: _shift(*b) for k, b in self.sequence_map.items()}
        self.sequence_map[text_key] = (from_seq_id, new_to_seq_id,)

    def _update_separator_ids(self, remove_from_id: int, remove_to_id: int, shift_offset: int):
        old_ids = self.separator_seq_ids
        if len(old_ids) > 0:
            idx1 = bisect.bisect_right(old_ids, remove_from_id)
            idx2 = bisect.bisect_left(old_ids, remove_to_id)
            self.separator_seq_ids = old_ids[:idx1] + \
                                     (np.array(old_ids[idx2:], dtype=int) + shift_offset).tolist()

    def _split_separators(self, seq_from_id: int, seq_to_id: int) -> Tuple[List[int], List[int]]:
        old_ids = self.separator_seq_ids
        if len(old_ids) > 0:
            idx1 = bisect.bisect_right(old_ids, seq_from_id)
            idx2 = bisect.bisect_left(old_ids, seq_to_id)
            return old_ids[:idx1], old_ids[idx2:],
        else:
            return [], [],

    def _is_start_section(self):
        sep_ids = self.separator_seq_ids
        if len(sep_ids) > 0:
            last_sep_id = sep_ids[-1]
            if len(self.token_ids) + self.first_token_seq_id == last_sep_id:
                return True
        return False

    def add_separator(self):
        # self._pad_last_chunk(pad_token_id)
        # separator_seq_ids always assumed to be ordered
        self.separator_seq_ids.append(self.first_token_seq_id + len(self.token_ids))

    def add_sequence(self, new_token_ids: List[int], metadata: Optional[Any],
                     importance: Optional[float] = None, timestamp: Optional[float] = None,
                     _no_new_chunks: bool = False,
                     _no_text_key: bool = False,
                     _text_key: TextKeyType = None) -> Tuple[List[Chunk], TextKeyType]:
        """
        Adds tokens to the chunk queue.
        :param new_token_ids: The sequence of token IDs to add
        :param metadata: A metadata object
        :param importance: An optional importance value for the tokens
        :param timestamp: The timestamp of the stored tokens
        :param _no_new_chunks: If true, attempt should be made to add all tokens without adding new chunks
        :param _no_text_key: Whether associating text with a key should be prevented.
        :param _text_key: A text key to be used instead of a generated one.
        :return: Any chunks removed due to overflow.
        """
        if len(new_token_ids) == 0:
            return [], _text_key,
        if timestamp is None:
            timestamp = time.time()
        text_key = None if _no_text_key else (_text_key if _text_key is not None else self._new_text_key())
        first_id = self.first_token_seq_id
        prev_token_seq_id = len(self.token_ids) + first_id
        starts_section = self._is_start_section()
        self.token_ids.extend(new_token_ids)
        next_token_seq_id = len(self.token_ids) + first_id
        start_num_chunks = len(self.chunks)
        first_c_idx = max(0, start_num_chunks if starts_section else start_num_chunks - 2)
        key_registered = False
        new_chunk_starts_section = starts_section
        for c_idx in range(first_c_idx, start_num_chunks + len(new_token_ids) + 1):
            if _no_new_chunks:
                if c_idx >= len(self.chunks):
                    raise RuntimeError('No new chunks allowed, but at least one more is needed to complete operation')
            else:
                while c_idx >= len(self.chunks):
                    self.add_chunk(metadata, importance, timestamp=timestamp, starts_section=new_chunk_starts_section,
                                   section_seq_id=prev_token_seq_id)
                    new_chunk_starts_section = False
            chunk = self.chunks[c_idx]
            if chunk.to_token_seq_id < next_token_seq_id:
                room = chunk.get_room()
                if room > 0:
                    if not _no_text_key:
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
        return self.check_overflow(), text_key,

    def _split_left(self, head_chunks: List[Chunk], tail_id_from: int):
        first_id = self.first_token_seq_id
        if len(head_chunks) > 0:
            last_head_chunk = head_chunks[-1]
            last_head_chunk_to = last_head_chunk.to_token_seq_id
            left_extras_from = last_head_chunk_to - first_id
            left_extras_to = tail_id_from - first_id
            left_extras_token_ids = self.token_ids[left_extras_from:left_extras_to]
            head_token_ids = self.token_ids[:left_extras_from]
            chunk_params = dict(
                timestamp=last_head_chunk.timestamp,
                importance=last_head_chunk.importance,
                metadata=last_head_chunk.metadata,
            )
        else:
            left_extras_token_ids = []
            head_token_ids = self.token_ids[:tail_id_from - first_id]
            chunk_params = dict(metadata=None)
        return head_token_ids, left_extras_token_ids, chunk_params,

    def _split_right(self, shifted_chunks: List[Chunk], head_id_to: int):
        first_id = self.first_token_seq_id
        if len(shifted_chunks) > 0:
            first_shifted_chunk = shifted_chunks[0]
            first_shifted_chunk_from = first_shifted_chunk.from_token_seq_id
            right_extras_from = head_id_to - first_id
            right_extras_to = first_shifted_chunk_from - first_id
            right_extras_token_ids = self.token_ids[right_extras_from:right_extras_to]
            shifted_token_ids = self.token_ids[right_extras_to:]
            chunk_params = dict(
                timestamp=first_shifted_chunk.timestamp,
                importance=first_shifted_chunk.importance,
                metadata=first_shifted_chunk.metadata,
            )
        else:
            right_extras_token_ids = []
            shifted_token_ids = self.token_ids[head_id_to:]
            chunk_params = dict(metadata=None)
        return shifted_token_ids, right_extras_token_ids, chunk_params,

    def _resolve_discarded_chunks(self, discarded_chunks: List[Chunk], text_key: TextKeyType,
                                  seq_id_from: int, old_seq_id_to: int, new_seq_id_to: int,
                                  shift_offset: int):
        # Remove chunks from map, etc.
        for chunk in discarded_chunks:
            self._removed_chunk_cleanup(chunk)
        # Update the bounds of the current sequence, and shift the ones to the right
        self._update_sequence_map(text_key, new_seq_id_to, shift_offset)
        # Remove and shift separator seq IDs
        self._update_separator_ids(seq_id_from, old_seq_id_to, shift_offset)
        return discarded_chunks

    def replace_sequence(self, text_key: TextKeyType, new_token_ids: List[int],
                         metadata: Optional[dict] = None, importance: Optional[float] = None,
                         timestamp: Optional[float] = None) -> Tuple[List[Chunk], TextKeyType]:
        old_bounds = self.sequence_map.get(text_key)
        if old_bounds is None:
            logging.warning(f'Subsequence with key {text_key} not found.')
            return [], text_key,
        seq_id_from, old_seq_id_to = old_bounds
        new_seq_id_to = seq_id_from + len(new_token_ids)
        discarded_chunks = []
        shifted_chunks = []
        head_chunks = []
        for chunk in self.chunks:
            if chunk.to_token_seq_id > seq_id_from:
                if chunk.from_token_seq_id >= old_seq_id_to:
                    shifted_chunks.append(chunk)
                else:
                    discarded_chunks.append(chunk)
            else:
                head_chunks.append(chunk)
        # Find out which tokens will be left out prior to first shifted chunk, plus overlap
        shifted_token_ids, right_extras_token_ids, right_params = self._split_right(shifted_chunks, old_seq_id_to)
        # Find out which tokens will be left out after the last head chunk, before the new sequence
        head_token_ids, left_extras_token_ids, left_params = self._split_left(head_chunks, seq_id_from)
        # Split separator IDs
        head_sep_ids, shifted_sep_ids = self._split_separators(seq_id_from, old_seq_id_to)
        # Truncate chunks, token IDs and separators
        self.token_ids = head_token_ids
        self.chunks = head_chunks
        self.separator_seq_ids = head_sep_ids
        # Add new token sequence to truncated queues
        self.add_sequence(left_extras_token_ids, _no_text_key=True, **left_params)
        self.add_sequence(new_token_ids, metadata=metadata, importance=importance,
                          timestamp=timestamp)
        self.add_sequence(right_extras_token_ids, _no_text_key=True, **right_params)
        # Calculate the shift offset
        shift_offset = new_seq_id_to - old_seq_id_to
        # Shift the sequence IDs of the chunks that need to be shifted
        # No padding is added, so shifted chunks should contain the same tokens as before
        for chunk in shifted_chunks:
            chunk.shift(shift_offset)
        # Extend the token IDs queue and the chunk queue
        self.token_ids.extend(shifted_token_ids)
        self.chunks.extend(shifted_chunks)
        self.separator_seq_ids.extend(shifted_sep_ids)
        # This call updates separator seq IDs according to the shift offset
        dc1 = self._resolve_discarded_chunks(discarded_chunks, text_key,
                                             seq_id_from, old_seq_id_to, new_seq_id_to,
                                             shift_offset)
        dc2 = self.check_overflow()
        return dc1 + dc2, text_key,

    def get_sequence_token_ids(self, text_key: TextKeyType) -> Optional[List[int]]:
        seq_bounds = self.sequence_map.get(text_key)
        if seq_bounds is None:
            logging.warning(f'Subsequence with key {text_key} not found.')
            return None
        seq_id_from, seq_id_to = seq_bounds
        first_id = self.first_token_seq_id
        index_from, index_to = seq_id_from - first_id, seq_id_to - first_id,
        return self.token_ids[index_from:index_to]

    def get_chunks_for_indexing(self) -> Tuple[List[Chunk], List[List[int]]]:
        # TODO not super efficient
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
        self.separator_seq_ids = []
        self.first_token_seq_id = 0
        self.current_chunk_id = 0
        self.chunk_map.clear()
        self.sequence_map.clear()

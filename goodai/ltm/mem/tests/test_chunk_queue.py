import math
import unittest

from goodai.ltm.mem.chunk_queue import ChunkQueue, ChunkExpansionOptions


class TestChunkQueue(unittest.TestCase):
    def test_insertion(self):
        seq_len = 8
        queue_capacity = 10
        num_seqs_in_chunk = 3
        chunk_capacity = seq_len * num_seqs_in_chunk
        chunk_index_at_overlap = chunk_capacity // 2
        sq = ChunkQueue(queue_capacity, chunk_capacity, chunk_index_at_overlap)
        for i in range(queue_capacity * 3):
            for j in range(num_seqs_in_chunk):
                start = i * chunk_capacity + j * seq_len
                token_ids = list(range(start, start + seq_len))
                sq.add_sequence(token_ids, None)
        assert sq.get_queue_size() == queue_capacity
        last_token_id = queue_capacity * 3 * chunk_capacity - 1
        sqs = sq.get_queue_size()
        stored_token_ids = []
        for i in range(0, sqs, 2):
            chunk = sq.chunks[i]
            observed_chunk = sq.get_chunk_token_ids(chunk)
            stored_token_ids.extend(observed_chunk)
        assert stored_token_ids[-1] == last_token_id
        for i in range(1, len(stored_token_ids)):
            assert stored_token_ids[i] == stored_token_ids[i-1] + 1
        stored_token_ids = []
        for i in range(1, sqs, 2):
            chunk = sq.chunks[i]
            observed_chunk = sq.get_chunk_token_ids(chunk)
            stored_token_ids.extend(observed_chunk)
        for i in range(1, len(stored_token_ids)):
            assert stored_token_ids[i] == stored_token_ids[i-1] + 1

    def test_full_capacity_insertion(self):
        queue_capacity = 10
        chunk_capacity = 24
        chunk_index_at_overlap = chunk_capacity // 2
        sq = ChunkQueue(queue_capacity, chunk_capacity, chunk_index_at_overlap)
        sequence = list(range(chunk_capacity * 3))
        sq.add_sequence(sequence, None)
        sqs = sq.get_queue_size()
        stored_token_ids = []
        for i in range(0, sqs, 2):
            chunk = sq.chunks[i]
            observed_chunk = sq.get_chunk_token_ids(chunk)
            stored_token_ids.extend(observed_chunk)
        assert stored_token_ids == sequence

    def test_retrieve_complete_sequences_adds_tokens_from_adjacent_chunks(self):
        chunk_capacity = 5
        chunk_index_at_overlap = chunk_capacity // 2
        _chunk_queue = ChunkQueue(12, chunk_capacity, chunk_index_at_overlap)
        _chunk_queue.add_sequence([1, 2, 3], None)
        _chunk_queue.add_sequence([4, 5, 6], None)
        _chunk_queue.add_sequence([7, 8, 9], None)
        _chunk_queue.add_sequence([10, 11, 12], None)
        _chunk_queue.add_sequence([10, 8, 12], None)
        _chunk_queue.add_sequence([14, 11, 12], None)
        _chunk_queue.add_sequence([3, 8, 21], None)
        _chunk_queue.add_sequence([13, 12, 9], None)
        _chunk_queue.add_sequence([3, 4, 28], None)
        _chunk_queue.add_sequence([12, 8, 25], None)

        punctuation_ids = {3, 8}
        ce_options = ChunkExpansionOptions.default(chunk_capacity, punctuation_ids)

        chunk_ids = [3, 5]
        result = _chunk_queue.retrieve_complete_sequences(chunk_ids, ce_options)

        self.assertEqual(2, len(result)), f'got {len(result)} sequences'
        seq1 = result[0]
        seq2 = result[1]

        self.assertEqual([7, 8, 9, 10, 11, 12, 10, 8], seq1)
        self.assertEqual([9, 10, 11, 12, 10, 8, 12, 14, 11, 12, 3], seq2)

    def test_zero_chunk_overlap(self):
        chunk_capacity = 5
        chunk_index_at_overlap = chunk_capacity
        _chunk_queue = ChunkQueue(25, chunk_capacity, chunk_index_at_overlap, first_token_seq_id=17)
        _chunk_queue.add_sequence([1, 2, 3], None)
        _chunk_queue.add_sequence([4, 5, 6], None)
        _chunk_queue.add_sequence([7, 8, 9], None)
        _chunk_queue.add_sequence([10, 11, 12], None)
        _chunk_queue.add_sequence([10, 8, 12], None)
        _chunk_queue.add_sequence([14, 11, 12], None)
        _chunk_queue.add_sequence([3, 8, 21], None)
        _chunk_queue.add_sequence([13, 12, 9], None)
        _chunk_queue.add_sequence([3, 4, 28], None)
        _chunk_queue.add_sequence([12, 8, 25], None)

        punctuation_ids = {3, 8}
        ce_options = ChunkExpansionOptions.default(chunk_capacity, punctuation_ids)

        chunk_ids = [3, 5]
        result = _chunk_queue.retrieve_complete_sequences(chunk_ids, ce_options)

        self.assertEqual(2, len(result)), f'got {len(result)} sequences'
        seq1 = result[0]
        seq2 = result[1]

        self.assertEqual([12, 14, 11, 12, 3, 8, 21, 13, 12, 9, 3], seq1)
        self.assertEqual([4, 28, 12, 8, 25], seq2)

    def test_overlap_of_1_token(self):
        chunk_capacity = 5
        chunk_index_at_overlap = chunk_capacity - 1
        _chunk_queue = ChunkQueue(25, chunk_capacity, chunk_index_at_overlap, first_token_seq_id=39)
        _chunk_queue.add_sequence([1, 2, 3], None)
        _chunk_queue.add_sequence([4, 5, 6], None)
        _chunk_queue.add_sequence([], None)
        _chunk_queue.add_sequence([7, 8, 9], None)
        _chunk_queue.add_sequence([10, 11, 12], None)
        _chunk_queue.add_sequence([10, 8, 12], None)
        _chunk_queue.add_sequence([], None)
        _chunk_queue.add_sequence([14, 11, 12], None)
        _chunk_queue.add_sequence([3, 8, 21], None)
        _chunk_queue.add_sequence([13, 12, 9], None)
        _chunk_queue.add_sequence([3, 4, 28], None)
        _chunk_queue.add_sequence([12, 8, 25], None)

        punctuation_ids = {3, 8}
        ce_options = ChunkExpansionOptions.default(chunk_capacity, punctuation_ids)

        chunk_ids = [3, 5]
        result = _chunk_queue.retrieve_complete_sequences(chunk_ids, ce_options)

        self.assertEqual(2, len(result)), f'got {len(result)} sequences'
        seq1 = result[0]
        seq2 = result[1]

        self.assertEqual([9, 10, 11, 12, 10, 8, 12, 14, 11, 12, 3], seq1)
        self.assertEqual([21, 13, 12, 9, 3, 4, 28, 12, 8], seq2)

    def test_separators(self):
        chunk_capacity = 5
        chunk_index_at_overlap = chunk_capacity // 2
        _chunk_queue = ChunkQueue(25, chunk_capacity, chunk_index_at_overlap, first_token_seq_id=75)
        _chunk_queue.add_sequence([1, 2, 3], None)
        _chunk_queue.add_sequence([4, 5, 6], None)
        _chunk_queue.add_separator(0)
        _chunk_queue.add_sequence([7, 8, 9], None)
        _chunk_queue.add_sequence([10, 11, 12], None)
        _chunk_queue.add_separator(0)
        _chunk_queue.add_sequence([10, 8, 12], None)
        _chunk_queue.add_sequence([14, 11, 12], None)
        _chunk_queue.add_separator(0)
        _chunk_queue.add_sequence([3, 8, 21], None)
        _chunk_queue.add_sequence([13, 12, 9], None)
        _chunk_queue.add_separator(0)
        _chunk_queue.add_sequence([3, 4, 28], None)
        _chunk_queue.add_sequence([12, 8, 25], None)

        punctuation_ids = {3, 8}
        ce_options = ChunkExpansionOptions.default(chunk_capacity, punctuation_ids)

        chunk_ids = [3, 6, 9]
        result = _chunk_queue.retrieve_complete_sequences(chunk_ids, ce_options)

        self.assertEqual(3, len(result)), f'got {len(result)} sequences'
        expected = [[7, 8, 9, 10, 11, 12, 0, 0, 0], [10, 8, 12, 14, 11, 12, 0, 0, 0], [3, 8, 21, 13, 12, 9, 0, 0, 0]]
        for e_seq, r_seq in zip(expected, result):
            self.assertEqual(e_seq, r_seq)

    def test_overflow_with_separators(self):
        chunk_capacity = 5
        queue_capacity = 10
        chunk_index_at_overlap = chunk_capacity // 2
        _chunk_queue = ChunkQueue(queue_capacity, chunk_capacity, chunk_index_at_overlap, first_token_seq_id=75)
        for i in range(200):
            _chunk_queue.add_sequence([1, 2, 3], None)
            _chunk_queue.add_separator(0)
        self.assertEqual(6, len(_chunk_queue.separator_seq_ids))
        self.assertTrue(_chunk_queue.separator_seq_ids[0] >= _chunk_queue.first_token_seq_id)
        self.assertTrue(_chunk_queue.separator_seq_ids[-1] <= _chunk_queue.first_token_seq_id +
                        len(_chunk_queue.token_ids))

    def test_flush(self):
        chunk_capacity = 5
        queue_capacity = 10
        chunk_index_at_overlap = chunk_capacity // 2
        _chunk_queue = ChunkQueue(queue_capacity, chunk_capacity, chunk_index_at_overlap, first_token_seq_id=220)
        for i in range(100):
            _chunk_queue.add_sequence([1, 2, 3], None)
            _chunk_queue.add_separator(0)
        _chunk_queue.flush()
        self.assertTrue(len(_chunk_queue.separator_seq_ids) == 0)
        self.assertTrue(len(_chunk_queue.chunks) == 0)
        self.assertTrue(len(_chunk_queue.token_ids) == 0)
        self.assertTrue(len(_chunk_queue.chunk_map) == 0)
        self.assertTrue(_chunk_queue.first_token_seq_id == 0)

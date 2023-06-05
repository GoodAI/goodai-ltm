import unittest
from goodai.ltm.mem.chunk_queue import ChunkQueue, ChunkExpansionOptions


class TestChunkQueue(unittest.TestCase):
    def test_simple_insertion(self):
        seq_len = 8
        queue_capacity = 10
        num_seqs_in_chunk = 3
        chunk_capacity = seq_len * num_seqs_in_chunk
        chunk_index_at_overlap = chunk_capacity // 2
        self.assertEqual(12, chunk_index_at_overlap)
        sq = ChunkQueue(queue_capacity, chunk_capacity, chunk_index_at_overlap)
        start = 0
        token_ids = list(range(start, start + seq_len))
        sq.add_sequence(token_ids, None)
        token_ids = list(range(start + seq_len, start + seq_len + seq_len))
        sq.add_sequence(token_ids, None)
        self.assertEqual(2, len(sq.chunks))
        self.assertEqual(16, len(sq.token_ids))

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
        _chunk_queue.add_separator()
        _chunk_queue.add_sequence([7, 8, 9], None)
        _chunk_queue.add_sequence([10, 11, 12], None)
        _chunk_queue.add_separator()
        _chunk_queue.add_sequence([10, 8, 12], None)
        _chunk_queue.add_sequence([14, 11, 12], None)
        _chunk_queue.add_separator()
        _chunk_queue.add_sequence([3, 8, 21], None)
        _chunk_queue.add_sequence([13, 12, 9], None)
        _chunk_queue.add_separator()
        _chunk_queue.add_sequence([3, 4, 28], None)
        _chunk_queue.add_sequence([12, 8, 25], None)

        punctuation_ids = {3, 8}
        ce_options = ChunkExpansionOptions.default(chunk_capacity, punctuation_ids)

        chunk_ids = [3, 6, 9]
        result = _chunk_queue.retrieve_complete_sequences(chunk_ids, ce_options)

        self.assertEqual(3, len(result)), f'got {len(result)} sequences'
        expected = [[7, 8, 9, 10, 11, 12], [10, 8, 12, 14, 11, 12], [3, 8, 21, 13, 12, 9]]
        for e_seq, r_seq in zip(expected, result):
            self.assertEqual(e_seq, r_seq)

    def test_overflow_with_separators(self):
        chunk_capacity = 5
        queue_capacity = 10
        chunk_index_at_overlap = chunk_capacity // 2
        _chunk_queue = ChunkQueue(queue_capacity, chunk_capacity, chunk_index_at_overlap, first_token_seq_id=75)
        for i in range(200):
            _chunk_queue.add_sequence([1, 2, 3], None)
            _chunk_queue.add_separator()
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
            _chunk_queue.add_separator()
        _chunk_queue.flush()
        self.assertTrue(len(_chunk_queue.separator_seq_ids) == 0)
        self.assertTrue(len(_chunk_queue.chunks) == 0)
        self.assertTrue(len(_chunk_queue.token_ids) == 0)
        self.assertTrue(len(_chunk_queue.chunk_map) == 0)
        self.assertTrue(_chunk_queue.first_token_seq_id == 0)

    def test_replacement_1(self):
        chunk_capacity = 5
        chunk_index_at_overlap = chunk_capacity // 2
        _chunk_queue = ChunkQueue(25, chunk_capacity, chunk_index_at_overlap, first_token_seq_id=73)
        _, k1 = _chunk_queue.add_sequence([1, 2, 3], None)
        _, k2 = _chunk_queue.add_sequence([4, 5, 6], None)
        _, k3 = _chunk_queue.add_sequence([7, 8, 9], None)

        chunk_sequences = _chunk_queue.get_chunk_sequences()
        expected_cs_1 = [[1, 2, 3, 4, 5], [3, 4, 5, 6, 7], [5, 6, 7, 8, 9], [7, 8, 9], [9]]
        self.assertEqual(expected_cs_1, chunk_sequences)

        _chunk_queue.replace_sequence(k2, [8, 9, 10, 9])
        chunk_sequences = _chunk_queue.get_chunk_sequences()
        expected_cs_2 = [[1, 2, 3, 8, 9], [3, 8, 9, 10, 9], [9, 10, 9], [9], [7, 8, 9], [9]]
        self.assertEqual(expected_cs_2, chunk_sequences)

    def test_replacement_2(self):
        chunk_capacity = 5
        chunk_index_at_overlap = chunk_capacity // 2
        _chunk_queue = ChunkQueue(25, chunk_capacity, chunk_index_at_overlap, first_token_seq_id=73)
        _chunk_queue.add_separator()
        _, k1 = _chunk_queue.add_sequence([1, 2, 3], None)
        _, k2 = _chunk_queue.add_sequence([4, 5, 6], None)
        _, k3 = _chunk_queue.add_sequence([7, 8, 9], None)

        chunk_sequences = _chunk_queue.get_chunk_sequences()
        expected_cs_1 = [[1, 2, 3, 4, 5], [3, 4, 5, 6, 7], [5, 6, 7, 8, 9], [7, 8, 9], [9]]
        self.assertEqual(expected_cs_1, chunk_sequences)

        _chunk_queue.replace_sequence(k1, [])
        chunk_sequences = _chunk_queue.get_chunk_sequences()
        expected_cs_2 = [[4], [5, 6, 7, 8, 9], [7, 8, 9], [9]]
        self.assertEqual(expected_cs_2, chunk_sequences)

    def test_replacement_3(self):
        chunk_capacity = 5
        chunk_index_at_overlap = chunk_capacity // 2
        _chunk_queue = ChunkQueue(25, chunk_capacity, chunk_index_at_overlap, first_token_seq_id=43)
        _, k1 = _chunk_queue.add_sequence([1, 2, 3], None)
        _, k2 = _chunk_queue.add_sequence([4, 5, 6], None)
        _chunk_queue.add_separator()
        _, k3 = _chunk_queue.add_sequence([7, 8, 9], None)

        chunk_sequences = _chunk_queue.get_chunk_sequences()
        expected_cs_1 = [[1, 2, 3, 4, 5], [3, 4, 5, 6], [5, 6], [7, 8, 9], [9]]
        self.assertEqual(expected_cs_1, chunk_sequences)

        _chunk_queue.replace_sequence(k3, [11, 12, 11, 12, 11])

        chunk_sequences = _chunk_queue.get_chunk_sequences()
        expected_cs_2 = [[1, 2, 3, 4, 5],
                         [3, 4, 5, 6],
                         [5, 6],
                         [11, 12, 11, 12, 11],
                         [11, 12, 11],
                         [11]]
        self.assertEqual(expected_cs_2, chunk_sequences)

        chunk_ids = [_chunk_queue.chunks[-4].chunk_id, _chunk_queue.chunks[-1].chunk_id]
        punctuation_ids = {90}
        ce_options = ChunkExpansionOptions.default(chunk_capacity, punctuation_ids)
        result = _chunk_queue.retrieve_complete_sequences(chunk_ids, ce_options)
        expected_result = [[1, 2, 3, 4, 5, 6], [11, 12, 11, 12, 11]]
        self.assertEqual(expected_result, result)

    def test_replacement_4(self):
        chunk_capacity = 5
        chunk_index_at_overlap = chunk_capacity // 2
        _chunk_queue = ChunkQueue(25, chunk_capacity, chunk_index_at_overlap, first_token_seq_id=43)
        _, k1 = _chunk_queue.add_sequence([1, 2, 3], None)
        _, k2 = _chunk_queue.add_sequence([4, 5, 6], None)
        _chunk_queue.add_separator()
        _, k3 = _chunk_queue.add_sequence([7, 8, 9], None)

        chunk_sequences = _chunk_queue.get_chunk_sequences()
        expected_cs_1 = [[1, 2, 3, 4, 5], [3, 4, 5, 6], [5, 6], [7, 8, 9], [9]]
        self.assertEqual(expected_cs_1, chunk_sequences)

        _chunk_queue.replace_sequence(k2, [11, 12, 13, 14, 15, 11, 12, 13, 14, 15])
        chunk_sequences = _chunk_queue.get_chunk_sequences()
        expected_cs_2 = [[1, 2, 3, 11, 12],
                         [3, 11, 12, 13, 14],
                         [12, 13, 14, 15, 11],
                         [14, 15, 11, 12, 13],
                         [11, 12, 13, 14, 15],
                         [13, 14, 15],
                         [15],
                         [7, 8, 9],
                         [9]]
        self.assertEqual(expected_cs_2, chunk_sequences)

        chunk_ids = [_chunk_queue.chunks[-4].chunk_id]
        ce_options = ChunkExpansionOptions(minSideTokens=0, maxSideTokens=100, leftStopAfterTokenIds=[],
                                           rightStopAtTokenIds=[])
        result = _chunk_queue.retrieve_complete_sequences(chunk_ids, ce_options)
        expected_result = [[1, 2, 3, 11, 12, 13, 14, 15, 11, 12, 13, 14, 15]]
        self.assertEqual(expected_result, result)

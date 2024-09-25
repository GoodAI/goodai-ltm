import pytest

from goodai.ltm.mem.chunk_queue import ChunkQueue


def test_chunkqueue_initialization():
    # Test valid initialization
    cq = ChunkQueue(capacity=5, chunk_capacity=10, chunk_index_at_overlap=6)
    assert cq.get_queue_size() == 0

    # Test invalid initializations
    with pytest.raises(ValueError):
        ChunkQueue(capacity=2, chunk_capacity=10, chunk_index_at_overlap=6)

    with pytest.raises(ValueError):
        ChunkQueue(capacity=5, chunk_capacity=0, chunk_index_at_overlap=6)

    with pytest.raises(ValueError):
        ChunkQueue(capacity=5, chunk_capacity=10, chunk_index_at_overlap=4)


def test_chunkqueue_add_chunk():
    cq = ChunkQueue(capacity=5, chunk_capacity=10, chunk_index_at_overlap=6)
    chunk = cq.add_chunk(metadata={"type": "test"}, importance=0.5, timestamp=123.456)
    assert chunk.chunk_id == 0
    assert chunk.capacity == 10
    assert chunk.metadata == {"type": "test"}
    assert chunk.importance == 0.5
    assert chunk.timestamp == 123.456


def test_chunkqueue_check_overflow():
    cq = ChunkQueue(capacity=3, chunk_capacity=10, chunk_index_at_overlap=6)
    for i in range(5):
        cq.add_chunk(metadata=None, importance=None, timestamp=123.456)
    removed_chunks = cq.check_overflow()
    assert len(removed_chunks) == 2
    assert cq.get_queue_size() == 3


def test_chunkqueue_add_sequence():
    cq = ChunkQueue(capacity=3, chunk_capacity=10, chunk_index_at_overlap=6)
    removed_chunks, text_key = cq.add_sequence([1, 2, 3, 4, 5], metadata=None, importance=None)
    assert len(removed_chunks) == 0
    assert text_key == 1
    assert cq.get_next_token_sequence_id() == 5


def test_chunkqueue_get_sequence_token_ids():
    cq = ChunkQueue(capacity=3, chunk_capacity=10, chunk_index_at_overlap=6)
    _, text_key = cq.add_sequence([1, 2, 3, 4, 5], metadata=None, importance=None)
    token_ids = cq.get_sequence_token_ids(text_key)
    assert token_ids == [1, 2, 3, 4, 5]


def test_chunkqueue_flush():
    cq = ChunkQueue(capacity=3, chunk_capacity=10, chunk_index_at_overlap=6)
    cq.add_sequence([1, 2, 3, 4, 5], metadata=None, importance=None)
    cq.flush()
    assert cq.get_queue_size() == 0
    assert cq.get_next_token_sequence_id() == 0

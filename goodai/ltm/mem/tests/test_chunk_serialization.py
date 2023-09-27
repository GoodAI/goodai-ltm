import pytest
from typing import Optional

from goodai.ltm.mem.chunk import Chunk


@pytest.fixture
def default_chunk():
    return Chunk(chunk_id=1, capacity=10, from_token_seq_id=0, metadata=None, importance=None, timestamp=123.456)


# Tests start here

def test_chunk_initialization(default_chunk):
    chunk = default_chunk
    assert chunk.chunk_id == 1
    assert chunk.capacity == 10
    assert chunk.from_token_seq_id == 0
    assert chunk.to_token_seq_id == 0
    assert chunk.indexed_length == -1
    assert chunk.timestamp == 123.456
    assert chunk.associated_keys == []


def test_chunk_length(default_chunk):
    chunk = default_chunk
    assert len(chunk) == 0

    chunk.to_token_seq_id = 5
    assert len(chunk) == 5


def test_chunk_room(default_chunk):
    chunk = default_chunk
    assert chunk.get_room() == 10

    chunk.to_token_seq_id = 5
    assert chunk.get_room() == 5


def test_chunk_at_capacity(default_chunk):
    chunk = default_chunk
    assert not chunk.is_at_capacity()

    chunk.to_token_seq_id = 10
    assert chunk.is_at_capacity()


def test_chunk_extend_by(default_chunk):
    chunk = default_chunk
    chunk.extend_by(5)
    assert chunk.to_token_seq_id == 5

    with pytest.raises(ValueError):
        chunk.extend_by(6)  # This should exceed capacity


def test_chunk_shift(default_chunk):
    chunk = default_chunk
    chunk.shift(5)
    assert chunk.from_token_seq_id == 5
    assert chunk.to_token_seq_id == 5


def test_chunk_indexed_state(default_chunk):
    chunk = default_chunk
    assert not chunk.is_indexed()

    chunk.update_indexed_state()
    assert chunk.is_indexed()


def test_chunk_add_key(default_chunk):
    chunk = default_chunk
    chunk.add_key(100)
    assert 100 in chunk.associated_keys


# Additional tests for Chunk creation and constructor

def test_chunk_creation_with_defaults():
    chunk = Chunk(chunk_id=2, capacity=20, from_token_seq_id=5, metadata=None, importance=None, timestamp=456.789)

    assert chunk.chunk_id == 2
    assert chunk.capacity == 20
    assert chunk.from_token_seq_id == 5
    assert chunk.to_token_seq_id == 5  # Should be initialized to from_token_seq_id
    assert chunk.indexed_length == -1
    assert chunk.timestamp == 456.789
    assert chunk.associated_keys == []
    assert chunk.importance is None
    assert chunk.metadata is None


def test_chunk_creation_with_metadata_and_importance():
    metadata = {"key": "value"}
    importance = 0.75
    chunk = Chunk(chunk_id=3, capacity=30, from_token_seq_id=10, metadata=metadata, importance=importance,
                  timestamp=789.012)

    assert chunk.chunk_id == 3
    assert chunk.capacity == 30
    assert chunk.from_token_seq_id == 10
    assert chunk.to_token_seq_id == 10
    assert chunk.indexed_length == -1
    assert chunk.timestamp == 789.012
    assert chunk.associated_keys == []
    assert chunk.importance == 0.75
    assert chunk.metadata == {"key": "value"}


def test_chunk_creation_with_defaults_no_keywords():
    chunk = Chunk(2, 20, 5, None, None, 456.789)

    assert chunk.chunk_id == 2
    assert chunk.capacity == 20
    assert chunk.from_token_seq_id == 5
    assert chunk.to_token_seq_id == 5  # Should be initialized to from_token_seq_id
    assert chunk.indexed_length == -1
    assert chunk.timestamp == 456.789
    assert chunk.associated_keys == []
    assert chunk.importance is None
    assert chunk.metadata is None


def test_chunk_creation_with_metadata_and_importance_no_keywords():
    metadata = {"key": "value"}
    importance = 0.75
    chunk = Chunk(3, 30, 10, metadata, importance, 789.012)

    assert chunk.chunk_id == 3
    assert chunk.capacity == 30
    assert chunk.from_token_seq_id == 10
    assert chunk.to_token_seq_id == 10
    assert chunk.indexed_length == -1
    assert chunk.timestamp == 789.012
    assert chunk.associated_keys == []
    assert chunk.importance == 0.75
    assert chunk.metadata == {"key": "value"}

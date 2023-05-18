from typing import Optional


class TextMemoryConfig:
    chunk_capacity: int
    """
    The capacity of a chunk, in tokens.
    """

    queue_capacity: int
    """
    The capacity of the memory's chunk queue, i.e. the maximum number of chunks it can hold.
    """

    reranking_k_factor: float
    """
    When a reranking mechanism is available, the value of k passed to the query function is
    multiplied by reranking_k_factor and the resulting number of chunks is retrieved from
    the chunk store before applying the reranking procedure.
    """

    max_query_length: Optional[int]
    """
    The maximum length of queries used to produce embeddings. The left side of queries
    exceeding this maximum length are truncated (after rewriting). 
    """

    chunk_overlap_fraction: float
    """
    The fraction of a chunk's length that overlaps with the previous and next chunk.
    This can be a value from 0 to 0.5.
    """

    redundancy_overlap_threshold: float
    """
    The fraction of a retrieved passage's length that causes it to be considered 
    redundant if it overlaps with other passages that are a better match to a query.
    """

    def __init__(self):
        self.max_query_length = 40  # Tokens
        self.chunk_capacity = 24  # Tokens
        self.queue_capacity = 5000  # Chunks
        self.reranking_k_factor = 10.0
        self.chunk_overlap_fraction = 0.5  # 0 to 0.5
        self.redundancy_overlap_threshold = 0.75

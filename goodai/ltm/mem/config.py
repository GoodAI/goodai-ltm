from typing import Optional, List


class ChunkExpansionConfig:
    def __init__(self, max_extra_side_tokens: int = 24,
                 left_stop_after: List[str] = None, right_stop_at: List[str] = None):
        if left_stop_after is None:
            left_stop_after = ['.', '!', '?']
        if right_stop_at is None:
            right_stop_at = ['.', '!', '?']
        self.max_extra_side_tokens = max_extra_side_tokens
        self.left_stop_after = left_stop_after
        self.right_stop_at = right_stop_at

    @classmethod
    def expand_to_sentence(cls, max_extra_side_tokens: int = 24):
        instance = cls(max_extra_side_tokens=max_extra_side_tokens)
        instance.left_stop_after = instance.right_stop_at = ['.', '!', '?']
        return instance

    @classmethod
    def expand_to_line_break(cls, max_extra_side_tokens: int = 64):
        instance = cls(max_extra_side_tokens=max_extra_side_tokens)
        instance.left_stop_after = instance.right_stop_at = ['\n', '\r\n']
        return instance

    @classmethod
    def expand_to_paragraph(cls, max_extra_side_tokens: int = 192):
        instance = cls(max_extra_side_tokens=max_extra_side_tokens)
        instance.left_stop_after = instance.right_stop_at = ['\n\n', '\r\n\r\n']
        return instance

    @classmethod
    def expand_to_section(cls, max_extra_side_tokens: int = 1024):
        instance = cls(max_extra_side_tokens=max_extra_side_tokens)
        instance.left_stop_after = instance.right_stop_at = []
        return instance


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

    chunk_expansion_config: ChunkExpansionConfig
    """
    The chunk expansion configuration
    """

    def __init__(self):
        self.max_query_length = 40  # Tokens
        self.chunk_capacity = 24  # Tokens
        self.queue_capacity = 64000  # Chunks
        self.reranking_k_factor = 10.0
        self.chunk_overlap_fraction = 0.5  # 0 to 0.5
        self.redundancy_overlap_threshold = 0.75
        self.chunk_expansion_config = ChunkExpansionConfig(max_extra_side_tokens=self.chunk_capacity)

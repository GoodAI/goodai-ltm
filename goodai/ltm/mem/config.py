class TextMemoryConfig:
    chunk_capacity: int
    """
    The capacity of a chunk, in tokens.
    """

    queue_capacity: int
    """
    The capacity of the memory's chunk queue, i.e. the maximum number of chunks it can hold.
    """

    reranking_k_factor: int
    """
    When a reranking mechanism is available, the value of k passed to the query function is
    multiplied by reranking_k_factor and the resulting number of chunks is retrieved from
    the chunk store before applying the reranking procedure.
    """

    def __init__(self):
        self.chunk_capacity = 24  # Tokens
        self.queue_capacity = 5000  # Chunks
        self.reranking_k_factor = 10

class TextMemoryConfig:
    def __init__(self):
        self.chunk_capacity = 24  # Tokens
        self.queue_capacity = 5000  # Chunks
        self.adjacent_chunks_ok = False

import pickle
from dataclasses import dataclass
from pathlib import Path

from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.default import DefaultTextMemory
from goodai.ltm.persistence.abstract import MemoryPersistence


@dataclass
class WholeMemoryPicklePersistence(MemoryPersistence):
    """
    Saves and loads the whole memory as a pickle file. Platform dependent and heavily version dependent.
    Saves embedding model too so it's huge.
    """
    pickle_memory_filename: str = "memory.pickle"

    def save(self, memory: DefaultTextMemory, directory: Path):
        file = directory / self.pickle_memory_filename
        with open(file, "bw") as fd:
            pickle.dump(memory, fd, protocol=pickle.DEFAULT_PROTOCOL)

    def load(self, directory: Path, **kwargs) -> DefaultTextMemory:
        file = directory / self.pickle_memory_filename
        with open(file, "br") as fd:
            result = pickle.load(fd)
            if not isinstance(result, DefaultTextMemory):
                raise ValueError(f"Expected BaseTextMemory, got {type(result)}")
            return result


@dataclass
class TargetedMemoryPicklePersistence(MemoryPersistence):
    """
    Saves and loads the memory in a targeted way as a pickle file. Platform dependent and heavily version dependent,
    but only takes as much space as necessary.
    """
    CHUNK_QUEUE_PICKLE_FILENAME = "chunk_queue.pickle"
    VECTOR_DB_PICKLE_FILENAME = "vector_db.pickle"

    def save(self, memory: DefaultTextMemory, directory: Path):
        chunk_queue = memory.chunk_queue
        file = directory / self.CHUNK_QUEUE_PICKLE_FILENAME
        with open(file, "bw") as fd:
            pickle.dump(chunk_queue, fd, protocol=pickle.DEFAULT_PROTOCOL)
        vector_db = memory.vector_db
        file = directory / self.VECTOR_DB_PICKLE_FILENAME
        with open(file, "bw") as fd:
            pickle.dump(vector_db, fd, protocol=pickle.DEFAULT_PROTOCOL)

    def load(self, directory: Path, **kwargs) -> DefaultTextMemory:
        file = directory / self.CHUNK_QUEUE_PICKLE_FILENAME
        with open(file, "br") as fd:
            chunk_queue = pickle.load(fd)
        file = directory / self.VECTOR_DB_PICKLE_FILENAME
        with open(file, "br") as fd:
            vector_db = pickle.load(fd)
        return AutoTextMemory.create(chunk_queue=chunk_queue, vector_db=vector_db, **kwargs)

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack

from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.default import DefaultTextMemory
from goodai.ltm.persistence.abstract import MemoryPersistence
from goodai.ltm.persistence.json_persistence import save_vector_db_to_npz, load_vector_db_from_npz, \
    load_chunk_queue_from_dict, obj_to_dict, load_vector_db_from_dict


def serialize_to_msgpack(obj: Any) -> bytes:
    return msgpack.packb(obj, default=obj_to_dict, use_bin_type=True)


@dataclass
class MsgPackMemoryPersistence(MemoryPersistence):
    """
    Saves and loads the whole memory as a msgpack file. Similar to json, but much more compact.
    """
    chunk_queue_filename: str = "chunk_queue.msgpack"
    vector_db_npz_filename = "vector_db.msgpack"

    def save(self, memory: DefaultTextMemory, directory: Path):
        chunk_queue = memory.chunk_queue
        file = directory / self.chunk_queue_filename
        with open(file, "wb") as fd:
            fd.write(serialize_to_msgpack(chunk_queue))
        vector_db = memory.vector_db
        file = directory / self.vector_db_npz_filename
        with open(file, "wb") as fd:
            fd.write(serialize_to_msgpack(vector_db))

    def load(self, directory: Path, **kwargs) -> DefaultTextMemory:
        file = directory / self.chunk_queue_filename
        with open(file, "rb") as data:
            chunk_queue = load_chunk_queue_from_dict(msgpack.unpackb(data.read(), raw=False, strict_map_key=False))

        file = directory / self.vector_db_npz_filename
        with open(file, "rb") as data:
            vector_db = load_vector_db_from_dict(msgpack.unpackb(data.read(), raw=False, strict_map_key=False))

        return AutoTextMemory.create(chunk_queue=chunk_queue, vector_db=vector_db, **kwargs)


@dataclass
class MsgNpzPackMemoryPersistence(MemoryPersistence):
    chunk_queue_filename: str = "chunk_queue.msgpack"
    vector_db_npz_filename = "vector_db.npz"

    def save(self, memory: DefaultTextMemory, directory: Path):
        chunk_queue = memory.chunk_queue
        file = directory / self.chunk_queue_filename
        with open(file, "wb") as fd:
            fd.write(serialize_to_msgpack(chunk_queue))
        vector_db = memory.vector_db
        file = directory / self.vector_db_npz_filename
        save_vector_db_to_npz(vector_db, file)

    def load(self, directory: Path, **kwargs) -> DefaultTextMemory:
        file = directory / self.chunk_queue_filename
        with open(file, "rb") as data:
            chunk_queue = load_chunk_queue_from_dict(msgpack.unpackb(data.read(), raw=False, strict_map_key=False))

        file = directory / self.vector_db_npz_filename
        vector_db = load_vector_db_from_npz(file)
        return AutoTextMemory.create(chunk_queue=chunk_queue, vector_db=vector_db, **kwargs)

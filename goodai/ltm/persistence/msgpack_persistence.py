from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import msgpack

from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.default import DefaultTextMemory
from goodai.ltm.persistence.abstract import MemoryPersistence
from goodai.ltm.persistence.json_persistence import save_vector_db_to_npz, load_vector_db_from_npz, \
    load_chunk_queue_from_dict, load_vector_db_from_dict, default_obj_to_dict, default_dict_to_obj


def serialize_to_msgpack(obj: Any, obj_to_dict: Callable) -> bytes:
    return msgpack.packb(obj, default=obj_to_dict, use_bin_type=True)


@dataclass
class MsgPackMemoryPersistence(MemoryPersistence):
    """
    Saves and loads the whole memory as a msgpack file. Similar to json, but much more compact.
    """
    chunk_queue_filename: str = "chunk_queue.msgpack"
    vector_db_npz_filename = "vector_db.msgpack"
    obj_to_dict: Callable = default_obj_to_dict
    dict_to_obj: Callable = default_dict_to_obj

    def save(self, memory: DefaultTextMemory, directory: Path):
        chunk_queue = memory.chunk_queue
        file = directory / self.chunk_queue_filename
        with open(file, "wb") as fd:
            fd.write(serialize_to_msgpack(chunk_queue, self.obj_to_dict))
        vector_db = memory.vector_db
        file = directory / self.vector_db_npz_filename
        with open(file, "wb") as fd:
            fd.write(serialize_to_msgpack(vector_db, self.obj_to_dict))

    def load(self, directory: Path, **kwargs) -> DefaultTextMemory:
        file = directory / self.chunk_queue_filename
        with open(file, "rb") as data:
            chunk_queue = load_chunk_queue_from_dict(
                msgpack.unpackb(data.read(), raw=False, strict_map_key=False, object_hook=self.dict_to_obj)
            )

        file = directory / self.vector_db_npz_filename
        with open(file, "rb") as data:
            vector_db = load_vector_db_from_dict(msgpack.unpackb(data.read(), raw=False, strict_map_key=False))

        return AutoTextMemory.create(chunk_queue=chunk_queue, vector_db=vector_db, **kwargs)

    def exists(self, directory: Path) -> bool:
        files = [
            directory / self.chunk_queue_filename,
            directory / self.vector_db_npz_filename,
        ]
        return all([file.exists() for file in files])


@dataclass
class MsgNpzPackMemoryPersistence(MemoryPersistence):
    chunk_queue_filename: str = "chunk_queue.msgpack"
    vector_db_npz_filename = "vector_db.npz"
    obj_to_dict: Callable = default_obj_to_dict
    dict_to_obj: Callable = default_dict_to_obj

    def save(self, memory: DefaultTextMemory, directory: Path):
        chunk_queue = memory.chunk_queue
        file = directory / self.chunk_queue_filename
        with open(file, "wb") as fd:
            fd.write(serialize_to_msgpack(chunk_queue, self.obj_to_dict))
        vector_db = memory.vector_db
        file = directory / self.vector_db_npz_filename
        save_vector_db_to_npz(vector_db, file)

    def load(self, directory: Path, **kwargs) -> DefaultTextMemory:
        file = directory / self.chunk_queue_filename
        with open(file, "rb") as data:
            chunk_queue = load_chunk_queue_from_dict(
                msgpack.unpackb(data.read(), raw=False, strict_map_key=False, object_hook=self.dict_to_obj)
            )

        file = directory / self.vector_db_npz_filename
        vector_db = load_vector_db_from_npz(file)
        return AutoTextMemory.create(chunk_queue=chunk_queue, vector_db=vector_db, **kwargs)

    def exists(self, directory: Path) -> bool:
        files = [
            directory / self.chunk_queue_filename,
            directory / self.vector_db_npz_filename,
        ]
        return all([file.exists() for file in files])

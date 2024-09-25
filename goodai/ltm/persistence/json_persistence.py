import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Callable

import numpy

from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.chunk import Chunk
from goodai.ltm.mem.chunk_queue import ChunkQueue
from goodai.ltm.mem.default import DefaultTextMemory
from goodai.ltm.mem.simple_vector_db import SimpleVectorDb
from goodai.ltm.persistence.abstract import MemoryPersistence


def default_obj_to_dict(obj):
    if isinstance(obj, numpy.ndarray):
        return obj.tolist()
    if isinstance(obj, datetime):
        return {"type": "datetime", "value": obj.isoformat()}
    if hasattr(obj, "__dict__"):
        obj_dict = obj.__dict__.copy()
        obj_dict.pop("chunk_map", None)
        return obj_dict
    raise TypeError(f"{type(obj)} is not JSON/Msgpack serializable")


def default_dict_to_obj(dct):
    if "type" in dct:
        if dct["type"] == "datetime":
            return datetime.fromisoformat(dct["value"])
    return dct


def serialize_to_json(obj: Any, obj_to_dict: Callable) -> str:
    return json.dumps(obj, default=obj_to_dict)


def save_vector_db_to_npz(vector_db: SimpleVectorDb, file: Path):
    numpy.savez(file, all_vectors=vector_db.all_vectors, all_ids=vector_db.all_ids)


def load_vector_db_from_npz(file: Path) -> SimpleVectorDb:
    with numpy.load(file) as data:
        all_vectors = data["all_vectors"]
        all_ids = data["all_ids"]
        vector_db = SimpleVectorDb(all_vectors=all_vectors, all_ids=all_ids)
    return vector_db


def save_vector_db_to_json(vector_db: SimpleVectorDb, file: Path, obj_to_dict: Callable = default_obj_to_dict):
    with open(file, "w", encoding="utf-8") as fd:
        fd.write(serialize_to_json(vector_db, obj_to_dict))


def load_vector_db_from_json(file: Path) -> SimpleVectorDb:
    with open(file, "r", encoding="utf-8") as fd:
        return load_vector_db_from_dict(json.load(fd))


def load_vector_db_from_dict(loaded_vector_db: Dict) -> SimpleVectorDb:
    return SimpleVectorDb(
        all_vectors=numpy.array(loaded_vector_db["all_vectors"]),
        all_ids=numpy.array(loaded_vector_db["all_ids"])
    )


def load_chunk_queue_from_dict(loaded_chunk_queue: Dict) -> ChunkQueue:
    typed_chunks = []
    for chunk in loaded_chunk_queue["chunks"]:
        c = Chunk(**chunk)
        typed_chunks.append(
            c
        )
    sequence_map = loaded_chunk_queue["sequence_map"]
    # json and msgpack don't allow string keys so it has to be converted back
    # values have to be converted from list to tuple
    loaded_chunk_queue["sequence_map"] = {int(key): tuple(value) for key, value in sequence_map.items()}
    loaded_chunk_queue["chunks"] = typed_chunks
    chunk_queue = ChunkQueue(**loaded_chunk_queue)
    return chunk_queue


@dataclass
class JsonMemoryPersistence(MemoryPersistence):
    """
    Saves and loads the memory to json files.
    Space inefficient since floats are stored as ascii characters.
    """
    chunk_queue_filename: str = "chunk_queue.json"
    vector_db_npz_filename = "vector_db.json"
    obj_to_dict: Callable = default_obj_to_dict
    dict_to_obj: Callable = default_dict_to_obj

    def save(self, memory: DefaultTextMemory, directory: Path):
        chunk_queue = memory.chunk_queue
        file = directory / self.chunk_queue_filename
        with open(file, "w", encoding="utf-8") as fd:
            fd.write(serialize_to_json(chunk_queue, self.obj_to_dict))

        vector_db = memory.vector_db
        file = directory / self.vector_db_npz_filename
        save_vector_db_to_json(vector_db, file)

    def load(self, directory: Path, **kwargs) -> DefaultTextMemory:
        file = directory / self.chunk_queue_filename
        chunk_queue = load_chunk_queue_from_dict(
            json.load(open(file, "r", encoding="utf-8"), object_hook=self.dict_to_obj)
        )

        file = directory / self.vector_db_npz_filename
        vector_db = load_vector_db_from_json(file)
        return AutoTextMemory.create(chunk_queue=chunk_queue, vector_db=vector_db, **kwargs)

    def exists(self, directory: Path) -> bool:
        files = [
            directory / self.chunk_queue_filename,
            directory / self.vector_db_npz_filename,
        ]
        return all([file.exists() for file in files])


@dataclass
class JsonNpzMemoryPersistence(MemoryPersistence):
    """
    Saves chunk queue to json and vector db to npz, which is space efficient.
    """
    chunk_queue_filename: str = "chunk_queue.json"
    vector_db_npz_filename = "vector_db.npz"
    obj_to_dict: Callable = default_obj_to_dict
    dict_to_obj: Callable = default_dict_to_obj

    def save(self, memory: DefaultTextMemory, directory: Path):
        chunk_queue = memory.chunk_queue
        file = directory / self.chunk_queue_filename
        with open(file, "w", encoding="utf-8") as fd:
            fd.write(serialize_to_json(chunk_queue, self.obj_to_dict))

        vector_db = memory.vector_db
        file = directory / self.vector_db_npz_filename
        save_vector_db_to_npz(vector_db, file)

    def load(self, directory: Path, **kwargs) -> DefaultTextMemory:
        file = directory / self.chunk_queue_filename
        chunk_queue = load_chunk_queue_from_dict(
            json.load(open(file, "r", encoding="utf-8"), object_hook=self.dict_to_obj))

        file = directory / self.vector_db_npz_filename
        vector_db = load_vector_db_from_npz(file)

        return AutoTextMemory.create(
            chunk_queue=chunk_queue, vector_db=vector_db,
            **kwargs
        )

    def exists(self, directory: Path) -> bool:
        files = [
            directory / self.chunk_queue_filename,
            directory / self.vector_db_npz_filename,
        ]
        return all([file.exists() for file in files])

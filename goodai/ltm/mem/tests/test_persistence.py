import tempfile
import unittest
from pathlib import Path

from goodai.ltm.embeddings.auto import AutoTextEmbeddingModel
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.base import BaseTextMemory
from goodai.ltm.mem.default import DefaultTextMemory
from goodai.ltm.persistence.abstract import MemoryPersistence
from goodai.ltm.persistence.json_persistence import JsonNpzMemoryPersistence, JsonMemoryPersistence
from goodai.ltm.persistence.msgpack_persistence import MsgNpzPackMemoryPersistence, MsgPackMemoryPersistence
from goodai.ltm.persistence.pickle_persistence import WholeMemoryPicklePersistence, TargetedMemoryPicklePersistence


class TestPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._lr_emb_model = AutoTextEmbeddingModel.shared_pretrained('em-MiniLM-p1-01')
        cls._text = "Earth has a dynamic atmosphere, which sustains Earth's surface conditions and protects " \
                    "it from most meteoroids and UV-light at entry. It has a composition of primarily nitrogen " \
                    "and oxygen. Water vapor is widely present in the atmosphere, forming clouds that cover most " \
                    "of the planet. The water vapor acts as a greenhouse gas and, together with other greenhouse " \
                    "gases in the atmosphere, particularly carbon dioxide (CO2), creates the conditions for both " \
                    "liquid surface water and water vapor to persist via the capturing of energy from the Sun's " \
                    "light. This process maintains the current average surface temperature of 14.76 °C, at " \
                    "which water is liquid under atmospheric pressure. Differences in the amount of captured energy " \
                    "between geographic regions (as with the equatorial region receiving more sunlight than the " \
                    "polar regions) drive atmospheric and ocean currents, producing a global climate system with " \
                    "different climate regions, and a range of weather phenomena such as precipitation, allowing " \
                    "components such as nitrogen to cycle."

        cls._kwargs = {
            "emb_model": cls._lr_emb_model
            # "emb_model": 'em-MiniLM-p1-01'
        }

    def test_saving_pickle(self):
        mem = self.do_saving(WholeMemoryPicklePersistence(), **self._kwargs)

    def test_saving_pickle_targeted(self):
        mem = self.do_saving(TargetedMemoryPicklePersistence(), **self._kwargs)

    def test_saving_json_npz(self):
        mem = self.do_saving(JsonNpzMemoryPersistence(), **self._kwargs)

    def test_saving_json(self):
        mem = self.do_saving(JsonMemoryPersistence(), **self._kwargs)

    def test_saving_msgpack_npz(self):
        mem = self.do_saving(MsgNpzPackMemoryPersistence(), **self._kwargs)

    def test_saving_msgpack(self):
        mem = self.do_saving(MsgPackMemoryPersistence(), **self._kwargs)

    def do_saving(self, memory_persistence: MemoryPersistence, **kwargs):
        mem: DefaultTextMemory = AutoTextMemory.create(**kwargs)
        mem.add_text(self._text, metadata={'foo': 'bar'})
        directory = tempfile.TemporaryDirectory().name
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        self.verify_memory(mem)

        memory_persistence.save(mem, path)

        loaded_memory = memory_persistence.load(path, **kwargs)

        assert loaded_memory.chunk_queue == mem.chunk_queue
        assert loaded_memory.vector_db == mem.vector_db

        self.verify_memory(loaded_memory)

        return loaded_memory

    def verify_memory(self, mem: BaseTextMemory):
        query = "Is water vapor widely present in the atmosphere?"
        r_memories = mem.retrieve(query, k=2)
        r_combined_text = '\n'.join([rm.passage for rm in r_memories])
        assert "Water vapor is widely present in the atmosphere" in r_combined_text

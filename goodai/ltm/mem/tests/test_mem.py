import gc
import unittest

from transformers import AutoTokenizer

from goodai.ltm.embeddings.auto import AutoTextEmbeddingModel
from goodai.ltm.eval.metrics import get_correctness_score
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.config import TextMemoryConfig


class TestMem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._lr_emb_model = AutoTextEmbeddingModel.from_pretrained('em-MiniLM-p3-01')
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

    def test_basic_usage(self):
        mem = AutoTextMemory.create(emb_model=self._lr_emb_model)
        mem.add_text(self._text, metadata={'foo': 'bar'})
        query = "Is water vapor widely present in the atmosphere?"
        r_memories = mem.retrieve(query, k=2)
        r_combined_text = '\n'.join([rm.passage for rm in r_memories])
        assert "Water vapor is widely present in the atmosphere" in r_combined_text
        m = r_memories[0].metadata
        assert m is not None and m.get('foo') == 'bar'

    def test_no_redundancy(self):
        config = TextMemoryConfig()
        config.redundancy_overlap_threshold = 0.5
        mem = AutoTextMemory.create(emb_model=self._lr_emb_model, config=config)
        mem.add_text(self._text, metadata={'foo': 'bar'})
        query = "Is water vapor widely present in the atmosphere? What forms clouds all over the planet?"
        r_memories = mem.retrieve(query, k=3)
        tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        for i in range(len(r_memories)):
            p1 = r_memories[i].passage
            for j in range(i + 1, len(r_memories)):
                p2 = r_memories[j].passage
                cs = get_correctness_score(tokenizer, p1, p2)
                assert cs <= 40

    def test_redundancy_allowed(self):
        config = TextMemoryConfig()
        config.redundancy_overlap_threshold = 1.0
        mem = AutoTextMemory.create(emb_model=self._lr_emb_model, config=config)
        mem.add_text(self._text, metadata={'foo': 'bar'})
        query = "Is water vapor widely present in the atmosphere? What forms clouds all over the planet?"
        r_memories = mem.retrieve(query, k=3)
        tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        match_count = 0
        for i in range(len(r_memories)):
            p1 = r_memories[i].passage
            for j in range(i + 1, len(r_memories)):
                p2 = r_memories[j].passage
                cs = get_correctness_score(tokenizer, p1, p2)
                if cs >= 50:
                    match_count += 1
        assert match_count > 0

    def test_multi_creation(self):
        # Should be able to create many default instances without running out of memory
        mems = []
        for i in range(100):
            mems.append(AutoTextMemory.create())

    def test_create_after_delete(self):
        mem1 = AutoTextMemory.create()
        mem1.add_text('foobar')
        del mem1
        gc.collect()
        mem2 = AutoTextMemory.create()
        mem2.add_text('foobar')

    def test_blank_query(self):
        mem1 = AutoTextMemory.create()
        mem1.add_text('foobar')
        mem1.retrieve('', k=5)

    def test_nearly_empty_mem(self):
        mem1 = AutoTextMemory.create()
        mem1.add_text('a')
        r_memories = mem1.retrieve('Is there something?', k=5)
        assert len(r_memories) > 0




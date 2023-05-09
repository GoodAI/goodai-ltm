import gc
import unittest

from goodai.ltm.embeddings.auto import AutoTextEmbeddingModel
from goodai.ltm.mem.auto import AutoTextMemory


class TestMem(unittest.TestCase):
    def test_basic_usage(self):
        emb_model = AutoTextEmbeddingModel.from_pretrained('em-MiniLM-p3-01')
        mem = AutoTextMemory.create(emb_model=emb_model)
        text = "Earth has a dynamic atmosphere, which sustains Earth's surface conditions and protects " \
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
        mem.add_text(text)
        query = "Is water vapor widely present in the atmosphere?"
        r_memories = mem.retrieve(query, k=2)
        r_combined_text = '\n'.join([rm.passage for rm in r_memories])
        assert "Water vapor is widely present in the atmosphere" in r_combined_text

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

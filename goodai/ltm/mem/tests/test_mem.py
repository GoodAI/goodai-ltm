import gc
import unittest
from typing import List, Tuple

from transformers import AutoTokenizer

from goodai.ltm.embeddings.auto import AutoTextEmbeddingModel
from goodai.ltm.eval.metrics import get_correctness_score
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.base import BaseReranker, RetrievedMemory, BaseTextMemory, BaseImportanceModel
from goodai.ltm.mem.config import TextMemoryConfig, ChunkExpansionConfig, ChunkExpansionLimitType
from goodai.ltm.reranking.base import BaseTextMatchingModel


class TestMem(unittest.TestCase):
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

    def test_multi_creation_with_name(self):
        # Should be able to create many default instances without running out of memory
        mems = []
        for i in range(100):
            mems.append(AutoTextMemory.create(emb_model='em-MiniLM-p1-01'))

    def test_multi_creation_with_qpm_model(self):
        # Should be able to create many default instances without running out of memory
        mems = []
        for i in range(100):
            mems.append(AutoTextMemory.create(emb_model='em-MiniLM-p1-01',
                                              matching_model='em:em-MiniLM-p1-01'))

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

    def test_custom_qpm_model(self):
        class _CustomQPM(BaseTextMatchingModel):
            def predict(self, sentences: List[Tuple[str, str]], batch_size: int = 32,
                        show_progress_bar: bool = False) -> List[float]:
                scores = []
                for query, passage in sentences:
                    if 'UV-light' in passage:
                        score = 0.9
                    elif 'greenhouse' in passage:
                        score = 0.7
                    else:
                        score = 0.5
                    scores.append(score)
                return scores

            def get_info(self):
                return ''

        qpm = _CustomQPM()
        mem = AutoTextMemory.create(matching_model=qpm, emb_model=self._lr_emb_model)
        mem.add_text(self._text)
        r_memories = mem.retrieve('Is water vapor widely present in the atmosphere?', k=5)
        assert 'UV-light' in r_memories[0].passage
        assert 'greenhouse' in r_memories[1].passage

    def test_separators_and_replacement(self):
        facts = [
            'Cane toads have a life expectancy of 10 to 15 years in the wild.',
            'Kayaks are used to transport people in water.',
            'Darth Vader is portrayed as a man who always appears in black full-body armor and a mask.',
            'Tony Bennett had four children.'
        ]
        config = TextMemoryConfig()
        config.chunk_capacity = 128
        mem = AutoTextMemory.create(emb_model=self._lr_emb_model, config=config)
        text_keys = []
        for i, fact in enumerate(facts):
            text_key = mem.add_text(fact, metadata={'index': i}, timestamp=i + 5)
            text_keys.append(text_key)
            mem.add_separator()
        facts[1] = "Higher education, also called post-secondary education, third-level or " \
                   "tertiary education, is an optional final stage of formal learning that " \
                   "occurs after completion of secondary education."
        mem.replace_text(text_keys[1], facts[1], metadata={'replacement': True}, timestamp=1 + 5)
        is_replacement = [False] * len(facts)
        is_replacement[1] = True
        for i, query in enumerate(facts):
            r_memories = mem.retrieve(query, k=1)
            self.assertEqual(query.strip(), r_memories[0].passage.strip())
            self.assertAlmostEqual(i + 5, r_memories[0].timestamp)
            if is_replacement[i]:
                self.assertTrue(r_memories[0].metadata['replacement'])
            else:
                self.assertEqual(i, r_memories[0].metadata['index'])

    def test_delete_all(self):
        facts = [
            'Cane toads have a life expectancy of 10 to 15 years in the wild.',
            'Kayaks are used to transport people in water.',
            'Darth Vader is portrayed as a man who always appears in black full-body armor and a mask.',
            'Tony Bennett had four children.'
        ]
        config = TextMemoryConfig()
        config.chunk_capacity = 128
        mem = AutoTextMemory.create(emb_model=self._lr_emb_model, config=config)
        text_keys = []
        for i, fact in enumerate(facts):
            text_key = mem.add_text(fact, metadata={'index': i}, timestamp=i + 5)
            text_keys.append(text_key)
            mem.add_separator()
        for tk in text_keys:
            mem.delete_text(tk)
        assert mem.is_empty()

    def test_expansion_to_paragraph(self):
        _text = "Earth has a dynamic atmosphere, which sustains Earth's surface conditions and protects " \
                "it from most meteoroids and UV-light at entry. It has a composition of primarily nitrogen " \
                "and oxygen. Water vapor is widely present in the atmosphere, forming clouds that cover most " \
                "of the planet. The water vapor acts as a greenhouse gas and, together with other greenhouse " \
                "gases in the atmosphere, particularly carbon dioxide (CO2), creates the conditions for both " \
                "liquid surface water and water vapor to persist via the capturing of energy from the Sun's " \
                "light.\n\n" \
                "This process maintains the current average surface temperature of 14.76 °C, at " \
                "which water is liquid under atmospheric pressure. Differences in the amount of captured energy " \
                "between geographic regions (as with the equatorial region receiving more sunlight than the " \
                "polar regions) drive atmospheric and ocean currents, producing a global climate system with " \
                "different climate regions, and a range of weather phenomena such as precipitation, allowing " \
                "components such as nitrogen to cycle."

        config = TextMemoryConfig()
        config.chunk_expansion_config = ChunkExpansionConfig.for_paragraph()
        mem = AutoTextMemory.create(emb_model=self._lr_emb_model, config=config)
        mem.add_text(_text)
        r1 = mem.retrieve("What is the composition of Earth's atmosphere?", k=1)[0]
        self.assertTrue(r1.passage.strip().startswith('Earth has a dynamic atmosphere'))
        self.assertTrue(r1.passage.strip().endswith("from the Sun's light."))
        r2 = mem.retrieve("What drives atmospheric and ocean currents?", k=1)[0]
        self.assertTrue(r2.passage.strip().startswith('This process maintains'))
        self.assertTrue(r2.passage.strip().endswith("such as nitrogen to cycle."))

    def test_expansion_to_lines(self):
        _text = "Earth has a dynamic atmosphere, which sustains Earth's surface conditions and protects " \
                "it from most meteoroids and UV-light at entry. It has a composition of primarily nitrogen " \
                "and oxygen.\n" \
                "Water vapor is widely present in the atmosphere, forming clouds that cover most " \
                "of the planet. The water vapor acts as a greenhouse gas and, together with other greenhouse " \
                "gases in the atmosphere, particularly carbon dioxide (CO2), creates the conditions for both " \
                "liquid surface water and water vapor to persist via the capturing of energy from the Sun's " \
                "light.\n" \
                "This process maintains the current average surface temperature of 14.76 °C, at " \
                "which water is liquid under atmospheric pressure. Differences in the amount of captured energy " \
                "between geographic regions (as with the equatorial region receiving more sunlight than the " \
                "polar regions) drive atmospheric and ocean currents, producing a global climate system with " \
                "different climate regions, and a range of weather phenomena such as precipitation,\n" \
                "allowing components such as nitrogen to cycle."

        config = TextMemoryConfig()
        config.chunk_expansion_config = ChunkExpansionConfig.for_line_break()
        config.chunk_expansion_config.min_extra_side_tokens = 0
        mem = AutoTextMemory.create(emb_model=self._lr_emb_model, config=config)
        mem.add_text(_text)
        r1 = mem.retrieve("Other than water vapor, what are other greenhouse gases?", k=1)[0]
        self.assertTrue(r1.passage.strip().startswith('Water vapor is widely present'))
        self.assertTrue(r1.passage.strip().endswith("from the Sun's light."))
        r2 = mem.retrieve("What drives atmospheric and ocean currents?", k=1)[0]
        self.assertTrue(r2.passage.strip().startswith('This process maintains'))
        self.assertTrue(r2.passage.strip().endswith("phenomena such as precipitation,"))

    def test_expansion_to_sections(self):
        config = TextMemoryConfig()
        config.chunk_expansion_config = ChunkExpansionConfig.for_section()
        mem = AutoTextMemory.create(emb_model=self._lr_emb_model, config=config)
        facts = [
            'Cane toads have a life expectancy of 10 to 15 years in the wild.',
            'Kayaks are used to transport people in water.',
        ]
        mem.add_text(facts[0])
        mem.add_separator()
        mem.add_text(self._text)
        mem.add_separator()
        mem.add_text(facts[1])
        r1 = mem.retrieve("Other than water vapor, what are other greenhouse gases?", k=1)[0]
        self.assertTrue(r1.passage.strip().startswith('Earth has a dynamic atmosphere'))
        self.assertTrue(r1.passage.strip().endswith("components such as nitrogen to cycle."))

    def test_excessive_chunk_expansion(self):
        cec = ChunkExpansionConfig(2048, limit_type=ChunkExpansionLimitType.SECTION)
        config = TextMemoryConfig()
        config.chunk_capacity = 12
        config.chunk_overlap_fraction = 0.5
        config.redundancy_overlap_threshold = 0.5
        config.chunk_expansion_config = cec
        with self.assertRaises(ValueError):
            AutoTextMemory.create(emb_model=self._lr_emb_model, config=config)

    def test_custom_reranker(self):
        class _LocalReranker(BaseReranker):
            def rerank(self, _r_memories: List[RetrievedMemory], _mem: BaseTextMemory) -> List[RetrievedMemory]:
                result = list(_r_memories)
                result.sort(key=lambda _m: _m.relevance)
                return result

        mem = AutoTextMemory.create(emb_model=self._lr_emb_model, reranker=_LocalReranker())
        mem.add_text(self._text)
        r_memories = mem.retrieve('Is water vapor widely present in the atmosphere?', k=5)
        self.assertIn('Water vapor is widely present', r_memories[-1].passage.strip())

    def test_custom_importance_model(self):
        class _LocalImportanceModel(BaseImportanceModel):
            def get_importance(self, mem_text: str):
                if 'kayaks' in mem_text.lower():
                    return 0.25
                elif 'vader' in mem_text.lower():
                    return 0.50
                else:
                    return 0

        facts = [
            'Cane toads have a life expectancy of 10 to 15 years in the wild.',
            'Kayaks are used to transport people in water.',
            'Darth Vader is portrayed as a man who always appears in black full-body armor and a mask.',
            'Tony Bennett had four children.'
        ]
        config = TextMemoryConfig()
        config.chunk_capacity = 128
        mem = AutoTextMemory.create(emb_model=self._lr_emb_model, importance_model=_LocalImportanceModel(),
                                    config=config)
        for i, fact in enumerate(facts):
            mem.add_text(fact, metadata={'index': i})
            mem.add_separator()

        r_memories = mem.retrieve('Generic question about anything.', k=5)
        for m in r_memories:
            p = m.passage.lower()
            if 'kayaks' in p:
                self.assertAlmostEqual(m.importance, 0.25)
            elif 'vader' in p:
                self.assertAlmostEqual(m.importance, 0.50)
            else:
                self.assertAlmostEqual(m.importance, 0)

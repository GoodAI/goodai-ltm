import unittest

from goodai.ltm.embeddings.auto import AutoTextEmbeddingModel
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.default import DefaultTextMemory


class TestMemStateAsText(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._lr_emb_model = AutoTextEmbeddingModel.shared_pretrained('em-MiniLM-p1-01')

    def test_restore_state_1(self):
        mem1 = AutoTextMemory.create(emb_model=self._lr_emb_model)
        mem1.add_text("foo")
        mem1.add_separator()
        mem1.add_text("bar")
        state1 = mem1.state_as_text()
        mem2: DefaultTextMemory = AutoTextMemory.create(emb_model=self._lr_emb_model)
        mem2.set_state(state1)
        state2 = mem2.state_as_text()
        self.assertEqual(state1, state2, "Mismatch of persisted state")
        chunk_queue = mem2.chunk_queue
        # Make sure chunk map got restored correctly
        for chunk in chunk_queue.chunks:
            found_chunk = chunk_queue.get_chunk(chunk.chunk_id)
            self.assertEqual(chunk, found_chunk)

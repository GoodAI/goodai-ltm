import time

from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.config import TextMemoryConfig, ChunkExpansionConfig
from goodai.ltm.reranking.stanford import StanfordReranker, StanfordImportanceModel

# This example uses a Stanford reranker equivalent to the one outlined
# in the generative agents paper: https://arxiv.org/pdf/2304.03442.pdf

_use_importance_model = False
_time = 0


def custom_timestamp() -> float:
    # This function is used instead of time.time() for memory timestamps.
    # Note that the StanfordReranker needs a time_fn parameter to use this.
    global _time
    _time += 1
    return _time


if __name__ == '__main__':
    # Note: For purposes of the example, the half_life is only 1.0 custom units,
    # but it should normally be measured in hours or days.
    reranker = StanfordReranker(half_life=1.0, use_importance=_use_importance_model,
                                time_fn=custom_timestamp)
    importance_model = StanfordImportanceModel() if _use_importance_model else None
    # Chunk expansion configuration appropriate for dialogue and event histories.
    config = TextMemoryConfig()
    config.chunk_expansion_config = ChunkExpansionConfig.for_line_break()
    config.redundancy_overlap_threshold = 0.5
    config.reranking_k_factor = 10
    mem = AutoTextMemory.create(reranker=reranker, importance_model=importance_model,
                                config=config)
    mem.add_text('Archie went to the well, where Jake is.\n',
                 timestamp=custom_timestamp())
    mem.add_text('Archie tells Jake: "Hey Jake, what\'s up?"\n',
                 timestamp=custom_timestamp())
    mem.add_text('Jake tells Archie: "Nothing much. What\'s up with you?"\n',
                 timestamp=custom_timestamp())
    mem.add_text('Archie tells Jake: "I heard you needed something from me?"\n',
                 timestamp=custom_timestamp())
    mem.add_text('Jake tells Archie: "Yes. Please get some nails from the hardware store."\n',
                 timestamp=custom_timestamp())
    mem.add_text('Archie tells Jake: "OK, I am heading over there now."\n',
                 timestamp=custom_timestamp())

    query = "Blake: Hey Archie, where did you go just now?\nArchie:"
    r_memories = mem.retrieve(query, k=2)
    for i, m in enumerate(r_memories):
        importance_text = f'{m.importance:.2g}' if m.importance is not None else 'N/A'
        print(f"Memory #{i+1} (relevance={m.relevance:.2g}, importance={importance_text}, ts={m.timestamp:.2g}):")
        print(m.passage.strip())
        print()

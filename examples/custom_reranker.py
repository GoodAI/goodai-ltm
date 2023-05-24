import time

from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.config import TextMemoryConfig, ChunkExpansionConfig
from goodai.ltm.reranking.stanford import StanfordReranker, StanfordImportanceModel

# This example uses a Stanford reranker equivalent to the one outlined
# in the generative agents paper: https://arxiv.org/pdf/2304.03442.pdf

if __name__ == '__main__':
    # Note: For purposes of the example, the half_life is only 0.3 seconds,
    # but it should normally be measured in hours.
    reranker = StanfordReranker(half_life=0.3)
    importance_model = StanfordImportanceModel()
    # Chunk expansion configuration appropriate for dialogue and event histories.
    config = TextMemoryConfig()
    config.chunk_expansion_config = ChunkExpansionConfig.for_line_break()
    config.redundancy_overlap_threshold = 0.5
    config.reranking_k_factor = 10
    mem = AutoTextMemory.create(reranker=reranker, importance_model=importance_model,
                                config=config)
    mem.add_text('Archie went to the well, where Jake is.\n')
    time.sleep(0.1)
    mem.add_text('Archie tells Jake: "Hey Jake, what\'s up?"\n')
    time.sleep(0.1)
    mem.add_text('Jake tells Archie: "Nothing much. What\'s up with you?"\n')
    time.sleep(0.1)
    mem.add_text('Archie tells Jake: "I heard you needed something from me?"\n')
    time.sleep(0.1)
    mem.add_text('Jake tells Archie: "Yes. Please get some nails from the hardware store."\n')
    time.sleep(0.1)
    mem.add_text('Archie tells Jake: "OK, I am heading over there now."\n')

    query = "Blake: Hey Archie, where did you go just now?\nArchie:"
    r_memories = mem.retrieve(query, k=2)
    for i, m in enumerate(r_memories):
        print(f"Memory #{i+1} (relevance={m.relevance:.2g}, importance={m.importance:.2g}):")
        print(m.passage.strip())
        print()

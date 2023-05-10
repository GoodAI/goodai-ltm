from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.rewrite_model import OpenAIRewriteModel

# This example demonstrates query and memory rewriting.


if __name__ == '__main__':
    # Create the rewrite model for queries and memories. An OpenAI API key needs to be in the
    # OPENAI_API_KEY environment variable.
    rewrite_model = OpenAIRewriteModel()

    # Test the rewrite model.
    query = "John: Not everyone is fond of ice cream. Mary: Do you like it?"
    print(f"Original query: {query}")
    query = rewrite_model.rewrite_query(query)
    print(f"Rewritten query: {query}")

    memory = ""

    # Create the memory.
    mem = AutoTextMemory.create(query_rewrite_model=rewrite_model, memory_rewrite_model=rewrite_model)


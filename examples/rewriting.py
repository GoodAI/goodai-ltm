from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.rewrite_model import OpenAIRewriteModel

# This example demonstrates query and memory rewriting

if __name__ == '__main__':
    # Create the rewrite model for queries and memories. An OpenAI API key needs to be in the OPENAI_API_KEY
    # environment variable. Model outputs can vary between calls with the same input. For deterministic outputs,
    # pass temperature=0.
    rewrite_model = OpenAIRewriteModel()

    # Test the rewrite model
    original_query = "John: Not everyone is fond of ice cream. Mary: Do you like it?"
    print(f"Original query: {original_query}")
    rewritten_query = rewrite_model.rewrite_query(original_query)
    print(f"Rewritten query: {rewritten_query}")

    original_query = "Archie is involved in some plan. What does he have in mind?"
    print(f"Original query: {original_query}")
    rewritten_query = rewrite_model.rewrite_query(original_query)
    print(f"Rewritten query: {rewritten_query}")

    context = """Jake Morales: Hey Archie, what do you think about teaming up with me and Isaac Coax? 
        We could come up with a plan that would distract Lucas Fern."""
    original_passage = "Archie: That would be great."
    print(f"Original passage: {original_passage}")
    rewritten_passage = rewrite_model.rewrite_memory(passage=original_passage, context=context)
    print(f"Rewritten passage: {rewritten_passage}")

    # Create the memory
    mem = AutoTextMemory.create(query_rewrite_model=rewrite_model, memory_rewrite_model=rewrite_model)

    # Use the memory with rewriting
    mem.add_text(text=original_passage, rewrite=True, rewrite_context=context)
    [retrieved_memory] = mem.retrieve(original_query, k=1, rewrite=True)
    print(retrieved_memory)

from goodai.ltm.mem.auto import AutoTextMemory

# This example adds some facts to the memory. Then we delete
# a specific fact. We also query the memory and delete a fact
# based on the query results.

if __name__ == '__main__':
    mem = AutoTextMemory.create(emb_model='em-distilroberta-p5-01')
    facts = [
        'Cane toads have a life expectancy of 10 to 15 years in the wild.',
        'Kayaks are used to transport people in water.',
        'Darth Vader is portrayed as a man who always appears in black full-body armor and a mask.',
        'Tony Bennett had four children.',
        'Higher education, also called post-secondary education, third-level or ' +
        'tertiary education, is an optional final stage of formal learning that ' +
        'occurs after completion of secondary education.'
    ]
    text_keys = []
    for fact in facts:
        if not mem.is_empty():
            mem.add_separator()
        tk = mem.add_text(fact)
        text_keys.append(tk)

    # Delete the second fact
    mem.delete_text(text_keys[1])

    # Query a passage and delete text snippets
    # associated with the matching chunks
    query1 = 'Who is Darth Vader?'
    r_memories = mem.retrieve(query1, k=1)
    for tk in r_memories[0].textKeys:
        mem.delete_text(tk)

    # Now try some queries:
    r_memories = mem.retrieve(query1, k=2)
    print(r_memories)

    query2 = 'What are Kayaks?'
    r_memories = mem.retrieve(query2, k=2)
    print(r_memories)




from goodai.ltm.mem.auto import AutoTextMemory
import wikipediaapi

# This example retrieves articles from wikipedia and stores them
# in the LTM, tagged with the article title

if __name__ == '__main__':
    mem = AutoTextMemory.create()
    wiki_wiki = wikipediaapi.Wikipedia('en')
    titles = [
        'Earth',
        'Python_(programming_language)',
    ]
    for article_title in titles:
        article_page = wiki_wiki.page(article_title)
        article_text = article_page.text
        mem.add_text(article_text, metadata={'title': article_title})
    queries = [
        "When did the Earth form?",
        "Who originally created the Python programming language?",
        "Describe the composition of Earth's atmosphere.",
        "How does the garbage collector work in Python?",
        "How did Earth's atmosphere form?",
        "What are the functional programming aspects of Python?",
    ]
    for query in queries:
        print(f'\n### Query: {query}')
        r_memories = mem.retrieve(query, k=5)
        for i, r_memory in enumerate(r_memories):
            m = r_memory.metadata
            title = '' if m is None else m['title']
            print(f'Memory #{i+1} (title={title}):')
            print(r_memory.passage)

    # Workaround for exception in wiki_wiki destructor
    wiki_wiki._session.adapters = {}

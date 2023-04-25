from goodai.ltm.mem.auto import AutoTextMemory
import wikipedia
import mwparserfromhell

# This example retrieves articles from wikipedia and stores them
# in the LTM, tagged with the article title

if __name__ == '__main__':
    mem = AutoTextMemory.create()
    wikipedia.set_lang('en')
    titles = [
        'Earth',
        'Python_(programming_language)',
    ]
    for article_title in titles:
        article_text = wikipedia.page(article_title).content
        parsed_text = mwparserfromhell.parse(article_text)
        plain_text = parsed_text.strip_code()
        mem.add_text(plain_text, metadata={'title': article_title})
    # query = "Describe the composition of Earth's atmosphere."
    query = "How does the garbage collector work in Python?"
    r_memories = mem.retrieve(query, k=5)
    for i, r_memory in enumerate(r_memories):
        m = r_memory.metadata
        title = '' if m is None else m['title']
        print(f'Memory #{i+1} (title={title}):')
        print(r_memory.passage)

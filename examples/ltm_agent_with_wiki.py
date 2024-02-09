import codecs
import json
import os.path

import litellm

from goodai.ltm.agent import LTMAgent, LTMAgentVariant
import wikipediaapi

# This example retrieves articles from wikipedia and stores them
# in the knowledge base of LTMAgent.

_log_count = 0


def _prompt_callback(session_id: str, label: str, context: list[dict], completion: str):
    # This is for logging prompts into the local data directory
    global _log_count
    _log_count += 1
    dir_name = f"data/{session_id}"
    os.makedirs(dir_name, exist_ok=True)
    prompt_file_path = os.path.join(dir_name, f"{label}-prompt-{_log_count}.json")
    with codecs.open(prompt_file_path, "w") as fd:
        json.dump(context, fd)
    completion_file_path = os.path.join(dir_name, f"{label}-completion-{_log_count}.txt")
    with codecs.open(completion_file_path, "w") as fd:
        fd.write(completion)


if __name__ == '__main__':
    # model here can be anything supported by litellm
    agent = LTMAgent(model="gpt-3.5-turbo",
                     max_prompt_size=3000,
                     max_completion_tokens=1024,
                     variant=LTMAgentVariant.SEMANTIC_ONLY,
                     prompt_callback=_prompt_callback)
    wiki_wiki = wikipediaapi.Wikipedia('en')
    titles = [
        'Earth',
        'Python_(programming_language)',
    ]
    for article_title in titles:
        article_page = wiki_wiki.page(article_title)
        article_text = article_page.text
        agent.add_knowledge(article_text)
    queries = [
        "When did the Earth form?",
        "Who originally created the Python programming language?",
        "Describe the composition of Earth's atmosphere.",
        "How does the garbage collector work in Python?",
        "How did Earth's atmosphere form?",
        "What are the functional programming aspects of Python?",
    ]
    try:
        for query in queries:
            print(f'\n# User: {query}')
            response = agent.reply(query)
            print(f"# Assistant: {response}")
        # Let's clear the message history
        agent.new_session()
        # Episodic memory test
        query = "When exactly did I ask you about the origin of the Python language?"
        print(f"# User: {query}")
        response = agent.reply(query)
        print(f"# Assistant: {response}")
    finally:
        # Workaround for exception in wiki_wiki destructor
        wiki_wiki._session.adapters = {}

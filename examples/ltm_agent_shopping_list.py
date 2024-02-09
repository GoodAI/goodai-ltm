import codecs
import json
import os.path

import litellm

from goodai.ltm.agent import LTMAgent, LTMAgentVariant
import wikipediaapi

# In this example we ask the agent to keep track of a shopping list, and then we test it.

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
    # This test requires a fairly capable model
    # variant = LTMAgentVariant.TEXT_SCRATCHPAD also works
    agent = LTMAgent(model="gpt-4-1106-preview",
                     max_prompt_size=3000,
                     variant=LTMAgentVariant.QG_JSON_USER_INFO,
                     prompt_callback=_prompt_callback)
    script = [
        "I need you to keep track of my shopping list",
        "Let's start with 5 cartons of milk and two cans of tuna",
        "Remove one carton of milk and add 4 heads of lettuce",
        "Actually, add another can of tuna and remove 1 head of lettuce",
        "Add five apples and 3 oranges",
        "On second thought, remove 3 of the apples",
    ]
    for line in script:
        print(f'\n# User: {line}')
        response = agent.reply(line)
        print(f"# Assistant: {response}")
    # Let's clear the message history
    agent.new_session()
    # Recall test
    query = "Can you tell me what's in my shopping list?"
    print(f"# User: {query}")
    response = agent.reply(query)
    print(f"# Assistant: {response}")

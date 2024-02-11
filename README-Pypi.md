## GoodAI-LTM

GoodAI-LTM equips agents with text-based long-term memory by combining essential components such as 
text embedding models, reranking, vector databases, memory and query rewriting, automatic chunking, 
chunk metadata, and chunk expansion. This package is specifically designed to offer a dialog-centric 
memory stream for social agents.

Additionally, GoodAI-LTM includes a conversational agent component (LTMAgent) for seamless 
integration into Python-based apps.

## Installation

    pip install goodai-ltm

## Usage of LTMAgent

Call the `reply` method of an `LTMAgent` instance to get a response from the agent.

    from goodai.ltm.agent import LTMAgent
    
    agent = LTMAgent(model="gpt-3.5-turbo")
    response = agent.reply("What can you tell me about yourself?")
    print(response)

The `model` parameter can be tha name of any model supported by the [litellm library](https://github.com/BerriAI/litellm).

A session history is maintained automatically by the agent. If you want to start a 
new session, call the `new_session` method.

    agent.new_session()
    print(f"Number of messages in session: {len(agent.session.message_history)}")    

The agent has a conversational memory and also a knowledge base. You can tell the agent
to store knowledge by invoking the `add_knowledge` method.

    agent.clear_knowledge()
    agent.add_knowledge("The user's birthday is February 10.")
    agent.add_knowledge("Refer to the user as 'boss'.")
    response = agent.reply("Today is February 10. I think this is an important date. Can you remind me?")
    print(response)

`LTMAgent` is a seamless RAG system. The [ltm_agent_with_wiki](./examples/ltm_agent_with_wiki.py) example 
shows how to add Wikipedia articles to the agent's knowledge base.

You can persist the agent's configuration and its memories/knowledge by obtaining
its state as a string via the `state_as_text` method.

    state_text = agent.state_as_text()
    # Persist state_text to secondary storage

To build an agent from state text, call the `from_state_text` method.

    agent2 = LTMAgent.from_state_text(state_text)

Note that this does not restore the conversation session. The persist the conversation session
call the `state_as_text` method of the session.

    from goodai.ltm.agent import LTMAgentSession
    
    session_state_text = agent.session.state_as_text()
    # session_state_text can be persisted in secondary storage
    # The session.session_id field can serve as an identifier of the persisted session
    # Now let's restore the session in agent2
    p_session = LTMAgentSession.from_state_text(session_state_text)
    agent2.use_session(p_session)

## Additional information

Visit the Github page: https://github.com/GoodAI/goodai-ltm

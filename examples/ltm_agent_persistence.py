from goodai.ltm.agent import LTMAgent, LTMAgentSession, LTMAgentConfig


# In this example we simulate saving agent memory and an agent session
# and then rebuilding an agent and the session.


def get_saved_agent_info() -> tuple[str, str]:
    # Let's use a lightweight embedding model
    _config = LTMAgentConfig()
    _config.emb_model = "em-MiniLM-p1-01"
    _agent = LTMAgent(model="gpt-3.5-turbo",
                      max_prompt_size=3000,
                      config=_config)
    # Let's add some knowledge
    _agent.add_knowledge("If the user asks you to perform a mathematical operation "
                         "write Typescript code instead and do not perform the calculation.")
    _agent.add_knowledge("Iterative functions are preferred over recursive functions.")
    # Add an interaction
    _agent.reply("What's the 37th number of a Fibonacci sequence "
                 "divided by the 45th one?")
    # Get the memory state of the agent
    _agent_state = _agent.state_as_text()
    # We also want to save the current session messages
    _session_state = _agent.session.state_as_text()
    return _agent_state, _session_state


if __name__ == '__main__':
    agent_state, session_state = get_saved_agent_info()
    print(f"Agent state size: {len(agent_state)/1024:.3g} Kchars")
    print(f"Session state size: {len(session_state)/1024:.3g} Kchars")
    # Let's first restore the agent
    agent = LTMAgent.from_state_text(agent_state)
    query = "What are your operational instructions regarding mathematical operations?"
    response = agent.reply(query)
    print(f"# Response 1: {response}")
    # Confirm that session has not been restored
    agent.clear_conversation_memory()
    query = "Have I asked a question that involves a mathematical operation?"
    response = agent.reply(query)
    print(f"# Response 2: {response}")
    # Restore the session
    session = LTMAgentSession.from_state_text(session_state)
    agent.use_session(session)
    # Now ask agent to confirm we're in the right session
    response = agent.reply(query)
    print(f"# Response 3: {response}")
    # Let's print the contents of the session
    print("# Session history follows:")
    for message in agent.session.message_history:
        print(f"{message.role}: {message.content}")

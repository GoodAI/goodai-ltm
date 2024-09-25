import datetime
import enum
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from json import JSONDecodeError
from typing import List, Callable, Optional, Any, Union

from litellm import completion, completion_cost

from goodai.helpers.json_helper import sanitize_and_parse_json, SimpleJSONEncoder, SimpleJSONDecoder
from goodai.ltm.mem.auto import AutoTextMemory
from goodai.ltm.mem.base import RetrievedMemory
from goodai.ltm.mem.config import ChunkExpansionConfig, TextMemoryConfig

import tiktoken

from goodai.ltm.prompts.chronological_ltm import cltm_template_queries_info
from goodai.ltm.prompts.scratchpad_ltm import s_ltm_template_queries_info

_logger = logging.getLogger("exp_agent")
_txt_re = re.compile(r"^.*```(?:txt)?(.*)```.*$", re.MULTILINE | re.DOTALL)
_log_prompts = os.environ.get("LTM_BENCH_PROMPT_LOGGING", "False").lower() in ["true", "yes", "1"]
_default_system_message = """
You are a helpful AI assistant with a long-term memory. Prior interactions with the user are tagged with a timestamp. Current time: {datetime}.
"""
_user_info_system_message = """
You are an expert in helping AI assistants manage their knowledge about a user and their 
operating environment.
"""
_scratchpad_system_message = """
You are an expert in helping AI assistants manage their knowledge about a user and their 
operating environment.
"""
_convo_excerpts_prefix = f"# The following are excerpts from the early part of the conversation "\
                         f"or prior conversations, in chronological order:\n\n"
_kb_excerpts_prefix = f"# The following are relevant excerpts from your knowledge base:\n\n"


def _default_time(session_id: str, line_index: int) -> float:
    return time.time()


class LTMAgentVariant(enum.Enum):
    """The type of memory used by LTMAgent."""

    SEMANTIC_ONLY = 0,
    """    
    A fast memory that relies only on semantic retrieval of chunks.
    """

    QG_JSON_USER_INFO = 1,
    """
    In addition to semantic retrieval, this memory has query generation and it also
    maintains a JSON object with information provided by the user.
    """

    TEXT_SCRATCHPAD = 2,
    """
    In addition to semantic retrieval, this memory maintains a text scratchpad
    with information provided by the user.
    """


@dataclass
class LTMAgentConfig:
    """
    Configuration of LTMAgent.
    """

    system_message: str = None
    """
    Text of the system prompt.
    """

    ctx_fraction_for_mem: float = 0.5
    """
    How much of the prompt should be reserved for memory excerpts.
    """

    emb_model: str = "flag:BAAI/bge-base-en-v1.5"
    """
    The name of the embedding model, as supported by goodai-ltm.
    """

    chunk_size: int = 32
    """
    The size of a chunk in the text memory.
    """

    chunk_queue_capacity: int = 50000
    """
    The capacity of the chunk queue, in number of chunks.
    """

    chunk_overlap_fraction: float = 0.5
    """
    How much chunks overlap with one another. This can be a number from 0 to 0.5.
    """

    redundancy_overlap_threshold: float = 0.6
    """
    For retrieved passages, or expanded chunks, what's the maximum overlap they can
    have with an already-retrieved chunk so they aren't removed.
    """

    llm_temperature: float = 0.01
    """
    The temperature for chat interaction LLM completions.
    """

    mem_temperature: float = 0.01
    """
    The temperature for internal LTM-related LLM completions.
    """

    timeout: Optional[int] = 300.0
    """
    The timeout of LLM requests.
    """


@dataclass
class Message:
    """
    A message in a chat session
    """

    role: str
    """
    This can be "assistant" or "user".
    """

    content: str
    """
    The text of the message.
    """

    timestamp: float
    """
    The timestamp of the message. Normally, this is seconds since Epoch.
    """

    def as_llm_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @property
    def is_user(self) -> bool:
        return self.role == "user"


class LTMAgent:
    """
    A conversational agent with a long-term memory.
    """
    def __init__(
        self,
        variant: LTMAgentVariant = LTMAgentVariant.SEMANTIC_ONLY,
        model: str = None,
        max_prompt_size: int = 4000,
        max_completion_tokens: Optional[float] = None,
        config: LTMAgentConfig = None,
        time_fn: Callable[[str, int], float] = _default_time,
        prompt_callback: Callable[[str, str, list[dict], str], Any] = None
    ):
        """
        LTMAgent initializer.
        :param variant: The type of LTM implementation.
        :param model: Any LLM model supported by the litellm library.
        :param max_prompt_size: The maximum size of the LLM prompt, in tokens.
        :param max_completion_tokens: The maximum size of the completion.
        :param config: Additional agent configuration parameters.
        :param time_fn: Optional function for message timestamps.
        :param prompt_callback: Optional function used to get information on prompts sent to the LLM.
        """
        super().__init__()
        if config is None:
            config = LTMAgentConfig()
        self.variant = variant
        self.prompt_callback = prompt_callback
        self.max_prompt_size = max_prompt_size
        self.max_completion_tokens = max_completion_tokens
        self.config = config
        self.mem_temperature = config.mem_temperature
        self.llm_temperature = config.llm_temperature
        self.overlap_threshold = config.redundancy_overlap_threshold
        self.ctx_fraction_for_mem = config.ctx_fraction_for_mem
        self.time_fn = time_fn
        self._session: Optional['LTMAgentSession'] = None
        self.system_message_template = config.system_message or _default_system_message
        self.user_info: dict = {}
        self.wm_scratchpad: str = ""
        self.model = model
        mem_config = TextMemoryConfig()
        mem_config.queue_capacity = config.chunk_queue_capacity
        mem_config.chunk_capacity = config.chunk_size
        mem_config.chunk_overlap_fraction = config.chunk_overlap_fraction
        mem_config.redundancy_overlap_threshold = config.redundancy_overlap_threshold
        mem_config.chunk_expansion_config = ChunkExpansionConfig.for_line_break(
            min_extra_side_tokens=config.chunk_size // 4,
            max_extra_side_tokens=config.chunk_size * 4
        )
        mem_config.reranking_k_factor = 10
        self.convo_mem = AutoTextMemory.create(emb_model=config.emb_model,
                                               config=mem_config)
        self.kb_mem = AutoTextMemory.create(emb_model=config.emb_model,
                                            config=mem_config)

    @staticmethod
    def _num_tokens_from_string(string: str, model="gpt-4"):
        """Returns the number of tokens in a text string."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(string))

    @classmethod
    def _context_token_counts(cls, messages: List[dict]):
        """Calculates the total number of tokens in a list of messages."""
        total_tokens = 0
        for message in messages:
            total_tokens += cls._num_tokens_from_string(message["content"])
        return total_tokens

    @property
    def session(self) -> 'LTMAgentSession':
        """
        :return: The current agent session.
        """
        if self._session is None:
            self.new_session()
        return self._session

    def new_session(self) -> 'LTMAgentSession':
        """
        Creates a new LTMAgentSession object and sets it as the current session.
        :return: The new session object.
        """
        session_id = str(uuid.uuid4())
        self._session = LTMAgentSession(session_id=session_id, m_history=[])
        if not self.convo_mem.is_empty():
            self.convo_mem.add_separator()
        return self._session

    def use_session(self, session: 'LTMAgentSession'):
        """
        Replaces the current session with the one provided.
        :param session: An instance of LTMAgentSession.
        """
        self._session = session

    def add_knowledge(self, text: str, with_separator: bool = True,
                      show_progress_bar: bool = False):
        """
        Adds text to the agent's knowledge base.
        :param text: A document or statement.
        :param with_separator: Whether to add a separator to the memory so chunks of the next document are separate.
        :param show_progress_bar: Whether to show a progress bar while indexing chunks.
        """
        self.kb_mem.add_text(text, show_progress_bar=show_progress_bar)
        if with_separator:
            self.kb_mem.add_separator()

    def state_as_text(self) -> str:
        """
        :return: A string representation of the content of the agent's memories (including
        embeddings and chunks) in addition to agent configuration information.
        Note that callback functions are not part of the provided state string.
        """
        convo_mem_state = self.convo_mem.state_as_text()
        kb_mem_state = self.kb_mem.state_as_text()
        state = dict(model=self.model,
                     max_prompt_size=self.max_prompt_size,
                     max_completion_tokens=self.max_completion_tokens,
                     config=self.config,
                     convo_mem=convo_mem_state,
                     kb_mem=kb_mem_state,
                     user_info=self.user_info,
                     wm_scratchpad=self.wm_scratchpad,
                     )
        return json.dumps(state, cls=SimpleJSONEncoder)

    @classmethod
    def from_state_text(cls, state_text: str,
                        time_fn: Callable[[str, int], float] = _default_time,
                        prompt_callback: Callable[
                            [str, str, list[dict], str], Any] = None) -> 'LTMAgent':
        """
        Builds an LTMAgent given a state string previously obtained by
        calling the state_as_text() method.
        :param state_text: A string previously obtained by calling the state_as_text() method.
        :param time_fn: Optional function for message timestamps.
        :param prompt_callback: Optional function used to get information on prompts sent to the LLM.
        :return:
        """
        state = json.loads(state_text, cls=SimpleJSONDecoder)
        model_name = state["model"]
        max_prompt_size = state["max_prompt_size"]
        max_completion_tokens = state["max_completion_tokens"]
        config = state["config"]
        convo_mem_state = state["convo_mem"]
        kb_mem_state = state["kb_mem"]
        user_info = state["user_info"]
        wm_scratchpad = state["wm_scratchpad"]
        agent = cls(max_prompt_size=max_prompt_size,
                    max_completion_tokens=max_completion_tokens,
                    model=model_name, config=config,
                    time_fn=time_fn, prompt_callback=prompt_callback)
        agent.convo_mem.set_state(convo_mem_state)
        agent.kb_mem.set_state(kb_mem_state)
        agent.user_info = user_info
        agent.wm_scratchpad = wm_scratchpad
        return agent

    def _prepare_system_content(self) -> str:
        new_system_content = ""
        if self.system_message_template:
            new_system_content = self.system_message_template.format(
                datetime=datetime.datetime.now())
        if self.variant == LTMAgentVariant.QG_JSON_USER_INFO:
            if self.user_info:
                user_info_text = json.dumps(self.user_info, indent=3)
                user_info_content = f"Current information about the user:\n{user_info_text}"
                if new_system_content:
                    new_system_content = new_system_content + '\n\n' + user_info_content
                else:
                    new_system_content = user_info_content
        elif self.variant == LTMAgentVariant.TEXT_SCRATCHPAD:
            if self.wm_scratchpad:
                user_info_content = f"Current scratchpad content (world model, user info):\n" \
                                    f"{self.wm_scratchpad}"
                if new_system_content:
                    new_system_content = new_system_content + "\n\n" + user_info_content
                else:
                    new_system_content = user_info_content
        return new_system_content

    def _update_info_object(self, info_object: Union[dict, str]):
        if self.variant == LTMAgentVariant.QG_JSON_USER_INFO:
            assert isinstance(info_object, dict)
            self.user_info = info_object
        elif self.variant == LTMAgentVariant.TEXT_SCRATCHPAD:
            assert isinstance(info_object, str)
            self.wm_scratchpad = info_object
        else:
            # nop
            pass

    def _build_llm_context(self, m_history: list[Message], user_content: str,
                           cost_callback: Callable[[float], Any]) -> list[dict]:
        target_history_tokens = self.max_prompt_size * (1.0 - self.ctx_fraction_for_mem)
        new_system_content = self._prepare_system_content()
        context = []
        if new_system_content:
            context.append({"role": "system", "content": new_system_content})
        context.append({"role": "user", "content": user_content})
        token_count = self._context_token_counts(context)
        to_timestamp = self.current_time
        for message in reversed(m_history):
            if message.is_user:
                et_descriptor = self._get_elapsed_time_descriptor(message.timestamp,
                                                                  self.current_time)
                new_content = f"[{et_descriptor}]\n{message.content}"
                message = Message(message.role, new_content, message.timestamp)
            message_dict = message.as_llm_dict()
            new_token_count = self._context_token_counts([message_dict]) + token_count
            if new_token_count > target_history_tokens:
                break
            to_timestamp = message.timestamp
            context.insert(1, message_dict)
            token_count = new_token_count
        remain_tokens = self.max_prompt_size - token_count
        info_object, mem_message = self._get_mem_message(m_history, user_content, remain_tokens,
                                                         to_timestamp=to_timestamp,
                                                         cost_callback=cost_callback)
        self._update_info_object(info_object)
        if mem_message:
            context.insert(1, mem_message)
        return context

    def convo_retrieve(self, queries: list[str], k_per_query: int,
                       to_timestamp: float) -> List[RetrievedMemory]:
        try:
            multi_list = self.convo_mem.retrieve_multiple(queries, k=k_per_query)
        except IndexError:
            _logger.error(f"Unable to retrieve memories using these queries: {queries}")
            raise
        r_memories = [rm for entry in multi_list for rm in entry]
        r_memories = [rm for rm in r_memories if rm.timestamp < to_timestamp]
        r_memories.sort(key=lambda _rm: _rm.relevance, reverse=True)
        r_memories = RetrievedMemory.remove_overlaps(r_memories,
                                                     overlap_threshold=self.overlap_threshold)
        return r_memories

    def kb_retrieve(self, queries: list[str], k_per_query: int) -> List[RetrievedMemory]:
        try:
            multi_list = self.kb_mem.retrieve_multiple(queries, k=k_per_query)
        except IndexError:
            _logger.error(f"Unable to retrieve memories using these queries: {queries}")
            raise
        r_memories = [rm for entry in multi_list for rm in entry]
        r_memories.sort(key=lambda _rm: _rm.relevance, reverse=True)
        r_memories = RetrievedMemory.remove_overlaps(r_memories,
                                                     overlap_threshold=self.overlap_threshold)
        return r_memories

    def _get_mem_message(
        self, m_history: list[Message], user_content: str, remain_tokens: int,
            to_timestamp: float, cost_callback: Callable[[float], Any], k_per_query=250
    ) -> tuple[Union[dict, str], Optional[dict[str, str]]]:
        """
        Gets (1) a new user object, and (2) a context message with
        information from memory if available.
        """
        queries, user_info = self._prepare_mem_info(m_history, user_content,
                                                    cost_callback=cost_callback)
        queries = queries or []
        queries = [f"user: {user_content}"] + queries
        convo_memories = self.convo_retrieve(queries, k_per_query=k_per_query,
                                             to_timestamp=to_timestamp)
        kb_memories = self.kb_retrieve(queries, k_per_query=k_per_query)
        excerpts_text = self.get_mem_excerpts(convo_memories, kb_memories,
                                              remain_tokens)
        if not excerpts_text:
            return user_info, None,
        return (
            user_info,
            dict(role="system", content=excerpts_text),
        )

    def _get_internal_prompt_system_message(self):
        if self.variant == LTMAgentVariant.QG_JSON_USER_INFO:
            return _user_info_system_message
        elif self.variant == LTMAgentVariant.TEXT_SCRATCHPAD:
            return _scratchpad_system_message
        else:
            raise ValueError(f"Unexpected: {self.variant}")

    @staticmethod
    def _sanitize_and_parse_scratchpad(s_completion: str) -> Optional[str]:
        match_txt = _txt_re.search(s_completion)
        if match_txt:
            return match_txt.group(1)
        else:
            _logger.warning(f"Scratchpad content not found in completion: {s_completion}")
            return None

    @staticmethod
    def _formatted_scratchpad(raw_scratchpad):
        return f"```txt\n{raw_scratchpad}\n```"

    def _complete_prepare_user_info(self, prompt_messages: list[dict],
                                    user_content: str,
                                    cost_callback: Callable[[float], Any]) -> tuple[list[str], dict]:
        if self.user_info:
            user_info_text = json.dumps(self.user_info, indent=3)
            user_info_description = f"Prior information about the user:\n{user_info_text}"
        else:
            user_info_description = f"There is no prior information about the user."
        sp_content = cltm_template_queries_info.format(
            user_info_description=user_info_description,
            user_content=user_content,
        ).strip()
        prompt_messages.append({"role": "user", "content": sp_content})
        query_json = self._completion(prompt_messages, temperature=self.mem_temperature,
                                      label="query-generation", cost_callback=cost_callback)
        try:
            queries_and_info = sanitize_and_parse_json(query_json)
        except (JSONDecodeError, ValueError):
            _logger.exception(f"Unable to parse JSON: {query_json}")
            queries_and_info = {}
        if not isinstance(queries_and_info, dict):
            _logger.warning("Query generation completion was not a dictionary!")
            queries_and_info = {}
        user_info = queries_and_info.get("user", self.user_info)
        _logger.info(f"New user object: {user_info}")
        queries = queries_and_info.get("queries", [])
        return (
            queries,
            user_info,
        )

    def _complete_prepare_wm_scratchpad(self, prompt_messages: list[dict],
                                        user_content: str,
                                        cost_callback: Callable[[float], Any]) -> str:
        if self.wm_scratchpad:
            user_info_description = (f"Prior scratchpad content (world model, user info):\n" +
                                     self._formatted_scratchpad(
                                         self.wm_scratchpad
                                     ))
        else:
            user_info_description = f"Prior scratchpad content is empty."
        sp_content = s_ltm_template_queries_info.format(
            user_info_description=user_info_description,
            user_content=user_content,
        ).strip()
        prompt_messages.append({"role": "user", "content": sp_content})
        s_completion = self._completion(prompt_messages, temperature=self.mem_temperature,
                                        label="scratchpad-generation",
                                        cost_callback=cost_callback)
        new_scratchpad = self._sanitize_and_parse_scratchpad(s_completion)
        return new_scratchpad

    def _prepare_mem_info(self, message_history: list[Message], user_content: str,
                          cost_callback: Callable[[float], Any]) -> tuple[list[str], Union[dict, str, None]]:
        if self.variant == LTMAgentVariant.SEMANTIC_ONLY:
            queries = [f"user: {user_content}"]
            return queries, None,
        system_message = self._get_internal_prompt_system_message()
        prompt_messages = [{"role": "system", "content": system_message}]
        last_assistant_message = None
        for i in range(len(message_history) - 1, -1, -1):
            m = message_history[i]
            if m.role == "assistant":
                last_assistant_message = m
                break
        if last_assistant_message:
            if len(message_history) > 2:
                prompt_messages.append({"role": "system",
                                        "content": "Prior conversation context omitted."})
            prompt_messages.append(last_assistant_message.as_llm_dict())
        if self.variant == LTMAgentVariant.QG_JSON_USER_INFO:
            return self._complete_prepare_user_info(prompt_messages, user_content, cost_callback)
        elif self.variant == LTMAgentVariant.TEXT_SCRATCHPAD:
            new_scratchpad = self._complete_prepare_wm_scratchpad(prompt_messages, user_content,
                                                                  cost_callback)
            queries = [f"user: {user_content}"]
            return queries, new_scratchpad,
        else:
            raise ValueError(f"Unexpected: {self.variant}")

    @staticmethod
    def _get_elapsed_time_descriptor(event_timestamp: float, current_timestamp: float):
        elapsed = current_timestamp - event_timestamp
        if elapsed < 1:
            return "just now"
        elif elapsed < 60:
            return f"{round(elapsed)} second(s) ago"
        elif elapsed < 60 * 60:
            return f"{round(elapsed / 60)} minute(s) ago"
        elif elapsed < 60 * 60 * 24:
            return f"{elapsed / (60 * 60):.1f} hour(s) ago"
        else:
            return f"{elapsed / (60 * 60 * 24):.1f} day(s) ago"

    def get_mem_excerpts(self, convo_memories: list[RetrievedMemory],
                         kb_memories: list[RetrievedMemory],
                         token_limit: int) -> str:
        all_memory_tuples: list[tuple[bool, RetrievedMemory]] = []
        all_memory_tuples.extend([(True, m,) for m in convo_memories])
        all_memory_tuples.extend([(False, m,) for m in kb_memories])
        all_memory_tuples.sort(key=lambda _t: _t[1].relevance, reverse=True)
        token_count = 0
        convo_excerpts: list[tuple[float, str]] = []
        kb_excerpts: list[tuple[float, str]] = []
        first_time = [True] * 2
        ts = self.current_time
        for is_convo, m in all_memory_tuples:
            if first_time[int(is_convo)]:
                first_time[int(is_convo)] = False
                prefix = _convo_excerpts_prefix if is_convo else _kb_excerpts_prefix
                new_token_count = self._num_tokens_from_string(prefix) + token_count
                if new_token_count > token_limit:
                    break
                token_count = new_token_count
            ts_descriptor = self._get_elapsed_time_descriptor(m.timestamp, current_timestamp=ts)
            if is_convo:
                excerpt = f"# Excerpt from {ts_descriptor}\n{m.passage.strip()}\n\n"
            else:
                excerpt = f"# Excerpt\n{m.passage.strip()}\n\n"
            new_token_count = self._num_tokens_from_string(excerpt) + token_count
            if new_token_count > token_limit:
                break
            token_count = new_token_count
            if is_convo:
                convo_excerpts.append(
                    (
                        m.timestamp,
                        excerpt,
                    )
                )
            else:
                kb_excerpts.append((m.passage_info.fromIndex, excerpt))
        convo_excerpts.sort(key=lambda _t: _t[0])
        kb_excerpts.sort(key=lambda _t: _t[0])
        result = ""
        if kb_excerpts:
            result += _kb_excerpts_prefix + "\n".join([e for _, e in kb_excerpts])
        if convo_excerpts:
            result += _convo_excerpts_prefix + "\n".join([e for _, e in convo_excerpts])
        return result

    @property
    def current_time(self) -> float:
        session = self.session
        session_id = session.session_id
        session_len = session.message_count
        return self.time_fn(session_id, session_len)

    def _add_to_convo_memory(self, message: "Message"):
        text = f"{message.role}: {message.content}\n"
        self.convo_mem.add_text(text, timestamp=message.timestamp)

    def reply(self, user_content: str, cost_callback: Callable[[float], Any] = None) -> str:
        """
        Asks the LLM to generate a completion from a user question/statement.
        This method first constructs a prompt from session history and memory excerpts.
        :param user_content: The user's question or statement.
        :param cost_callback: An optional function used to track LLM costs.
        :return: The agent's completion or reply.
        """
        session = self.session
        context = self._build_llm_context(session.message_history, user_content,
                                          cost_callback)
        response = self._completion(context, temperature=self.llm_temperature, label="reply",
                                    cost_callback=cost_callback)
        user_message = Message(role="user", content=user_content, timestamp=self.current_time)
        session.add(user_message)
        self._add_to_convo_memory(user_message)
        assistant_message = Message(role="assistant", content=response, timestamp=self.current_time)
        session.add(assistant_message)
        self._add_to_convo_memory(assistant_message)
        return response

    def _completion(self, context: List[dict[str, str]], temperature: float, label: str,
                    cost_callback: Callable[[float], Any]) -> str:
        response = completion(model=self.model, messages=context, timeout=self.config.timeout,
                              temperature=temperature, max_tokens=self.max_completion_tokens)
        response_text = response['choices'][0]['message']['content']
        if self.prompt_callback:
            self.prompt_callback(self.session.session_id, label, context, response_text)
        if cost_callback:
            cost = completion_cost(model=self.model, completion_response=response, messages=context)
            cost_callback(cost)
        return response_text

    def clear_knowledge(self):
        """
        Empties the agent's knowledge base.
        """
        self.kb_mem.clear()

    def clear_conversation_memory(self):
        """
        Empties the agent's conversation memory.
        """
        self.convo_mem.clear()
        self.user_info = {}
        self.wm_scratchpad = ""

    def reset(self):
        """
        Empties all agent memories and starts a new chat session.
        """
        self.clear_knowledge()
        self.clear_conversation_memory()
        self.new_session()


class LTMAgentSession:
    """
    An agent session, or a collection of messages.
    """
    def __init__(self, session_id: str, m_history: list[Message]):
        self.session_id = session_id
        self.message_history: list[Message] = m_history or []

    @property
    def message_count(self):
        return len(self.message_history)

    def state_as_text(self) -> str:
        """
        :return: A string that represents the contents of the session.
        """
        state = dict(session_id=self.session_id, history=self.message_history)
        return json.dumps(state, cls=SimpleJSONEncoder)

    def add(self, message: Message):
        self.message_history.append(message)

    @classmethod
    def from_state_text(cls, state_text: str) -> 'LTMAgentSession':
        """
        Builds a session object given state text.
        :param state_text: Text previously obtained using the state_as_text() method.
        :return: A session instance.
        """
        state: dict = json.loads(state_text, cls=SimpleJSONDecoder)
        session_id = state["session_id"]
        m_history = state["history"]
        return cls(session_id, m_history)

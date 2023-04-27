import json
import os
import string
from abc import abstractmethod, ABC
from textwrap import dedent

import openai


DEFAULT_QUERY_PROMPT_TEMPLATE = dedent("""\
    You will be provided a json document containing an "unprocessed" entry. 
    You will rewrite only the unprocessed entry into a query for a retrieval system. 
    Rewrite the unprocessed entry according to the instructions and examples below and store the rewritten query as a string called "processed". 
    - The entry contains a dialog with replies prefixed by the speaker's name. Keep track of the person making a request and the person receiving a request. Be careful to attribute requests to the right person. 
    { "unprocessed": "Joe: Do you know the password? Ben: No, I don't. But what I do need help with right now, if you're willing to lend a hand, has to do with Lucas Fern." } 
    { "processed": "Can Joe help Ben with Lucas Fern?" }
    - The entry contains replies prefixed by the speaker's name. Focus on the last entry.
    { "unprocessed": "Alice: So, I was thinking about how we can make this place more beautiful. Maybe you have some ideas? Gary: What is the password?" }
    { "processed": "What is the password?"}
    - Remove ellipsis, including pronouns and tacit information.
    { "unprocessed": "Jake Anderson: Have you seen the new F-150? James: Do you like it?"}
    { "processed": "Does Jake like the Ford F-150?" }
    - Adapt information based on qualifiers. Discard standalone qualifier sentences.
    { "unprocessed": "Ben: Tell me more. James: I am not sure about what I am to tell you next... Cows in Austria are violet."}
    { "processed": "James is not sure about it, but claims that cows in Austria are violet." }
    - Make subjective information objective. E.g.,
    { "unprocessed": "Emma: I like painting. What do you like? James: I think the best sport is cycling. I particularly enjoy Eddy Merckx."}
    { "processed": "James' favorite sport is cycling and James' favorite cyclist is Eddy Merckx." }
    - Wherever possible, include the answers to "who", "what", "when" and "where" in the processed . 
    { "unprocessed": "Nicholas: The Festival of the Stars is a yearly event that celebrates the celestial bodies and their influence on Aradia. Chris: It is a time of joy for all who attend!" }
    { "processed" : "Chris thinks the Festival of the Stars in Aradia is a time of joy for all who attend." }
    - Summarize wordy narratives into concise statements of relevant information.
    { "unprocessed": "Annie: What's the matter? John: I don't know where to begin... The guards are harassing me. It's awful... I don't know what to do. Just the other day it was very bad." }
    { "processed": "The guards are harassing John." }
    - When possible, summarize wordy entries into questions for relevant information.
    { "unprocessed": "Simon: Did you like the markets? Cecile: Yes, the seafood is good in Sicily and fresh vegetables are tasty and cheap. Why don't we live there... I could spend hours in those food markets... Do you remember the name of the old market in Palermo?"}
    { "processed": "What is the name of the old food market in Palermo?" }
    - Produce a complete sentence or query. Focus on the last reply but use information from preceding dialog if necessary.
    { "unprocessed": "Petr: Drones are cool. Jarda: Totally." }
    { "processed": "Jarda agrees that drones are cool." }
    - Don't ask questions that are answered in the dialog. Do ask questions for new information.
    { "unprocessed": "Ben: There is some food in the fridge. Jerry: I see. Will Johnny come over?" } 
    { "processed": "Will Johnny come over?" }

    Now rewrite the unprocessed entry. Keep track of the speakers in the dialog, but only mention them when necessary. 
    Don't ask questions that are answered in the dialog. Do ask for new information.
    { "unprocessed": "$query"}""")

DEFAULT_MEMORY_PROMPT_TEMPLATE = dedent("""\
    You will be provided a json document containing a "context" entry and an "unprocessed" entry. 
    You will rewrite only the unprocessed entry into individual facts that can be indexed easily by a retrieval system. 
    Rewrite the unprocessed entry according to the instructions below and store the rewritten facts as entries in a json array called "processed". 
    Make sure to split sentences into multiple rewritten facts if there is too much information in the "unprocessed" sentence.
    If you encounter a question, do not include it in the rewritten facts. Do not repeat facts or sentences mentioned in "context". Moreover:
    - remove ellipsis, including pronouns and tacit information.Example input/ output:
    { "context": "James: Have you seen the new F-150?\nJane: Sure, that thing is a beast, I wish I had it.\n",  "unprocessed": "James: I like it a lot."}
    { "processed": { "1": "James likes Ford F-150 a lot."} }
    - adapt information based on qualifiers. Discard standalone qualifier sentences.
    { "context": "James: I am not sure about what I am to tell you next.\nJane: Yes?\n", "unprocessed": "James: Cows in Austria are violet."}
    { "processed": { "1": "James is not sure about it, but claims that cows in Austria are violet."} }
    - split sentences with multiple unrelated pieces of information into separate, atomic facts. E.g.
    { "context": "Marry: What are your favorite things, James?", "unprocessed": "James: My favorite color is yellow and my favorite animal is elephant."}
    { "processed": { "1": "James' favorite color is yellow.", "2": "James' favorite animal is elephant."} }
    - make subjective information objective.E.g.
    { "context": "Jane: I love painting. It's the best thing.\n James: I like painting quite a bit too. But there is something else I like.\n", "unprocessed": "James: the best sport is cycling. I particularly enjoy Eddy Merckx."}
    { "processed": { "1": "James' favorite sport is cycling.", "2": "James' favorite cyclist is Eddy Merckx."} }
    - wherever possible, include the answers to "who", "what", "when" and "where" in the processed facts. 
    { "context": "There is a powerful and ancient magic that can bring people back from the dead, but it is forbidden and considered taboo by most inhabitants of Aradia.\nThe Festival of the Stars is a yearly event that celebrates the celestial bodies and their influence on Aradia.\n", "unprocessed": "It is a time of celebration and joy for all who attend." }
    { "processed" : { "1": "The Festival of the Stars in Aradia is a time of celebration and joy for all who attend."} }
    { "context": "$context" }
    { "unprocessed": "$passage" }""")

DEFAULT_QUERY_SYSTEM_MESSAGE = DEFAULT_MEMORY_SYSTEM_MESSAGE \
    = """You are a helpful assistant that rewrites queries and text passages for a text retrieval system."""


class BaseRewriteModel(ABC):
    """
    Abstract base class for query and memory rewrite models.

    Query and memory rewriting is used to disambiguate queries and passages and make them more self-contained.

    For example, the statement "It's the best!", referring to strawberry ice cream and uttered by John,
    can be rewritten as "John thinks that strawberry ice cream is the best."
    """

    @abstractmethod
    def rewrite_query(self, query: str) -> str:
        """
        Rewrite a query to disambiguate it and make it more self-contained.

        For example, the input

            John: People differ in their appreciation of ice cream.
            Mary: Do you like it?

        can be rewritten as

            Does John like ice cream?

        :param query: Query preceded by some context
        :return: Rewritten query
        """

        pass

    @abstractmethod
    def rewrite_memory(self, passage: str, context: str) -> str:
        """
        Rewrite a passage to disambiguate it and make it more self-contained.

        For example, the passage "Mary: I like it!" together with the context
        "People differ in their appreciation of ice cream." can be rewritten as "Mary likes ice cream".

        :param passage: The memory to be rewritten
        :param context: Context that helps disambiguate the memory
        :return: Rewritten memory
        """

        pass


class OpenAIRewriteModel(BaseRewriteModel):
    """
    Rewrite model wrapper for OpenAI completion models such as text-davinci-003.
    """

    def __init__(self,
                 model_name: str = "text-davinci-003",
                 api_key: str = None,
                 temperature: float = 0.7,
                 max_tokens: int = 512,
                 query_rewrite_prompt: str = None,
                 memory_rewrite_prompt: str = None):
        self.model_name = model_name
        if api_key:
            self.api_key = api_key
        elif 'OPENAI_API_KEY' in os.environ:
            self.api_key = os.environ['OPENAI_API_KEY']
        else:
            raise ValueError("An OpenAI api key needs to be provided as a parameter or environment variable")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.query_prompt_template = query_rewrite_prompt or DEFAULT_QUERY_PROMPT_TEMPLATE
        self.memory_prompt_template = memory_rewrite_prompt or DEFAULT_MEMORY_PROMPT_TEMPLATE

    def rewrite_query(self, query: str) -> str:
        prompt = self.make_query_prompt(query)
        return self.post_process_query(self.completion(prompt))

    def rewrite_memory(self, passage: str, context: str) -> str:
        prompt = self.make_memory_prompt(passage, context)
        return self.post_process_memory(self.completion(prompt))

    def completion(self, prompt):
        openai.api_key = self.api_key
        response = openai.Completion.create(model=self.model_name, prompt=prompt, temperature=self.temperature,
                                            max_tokens=self.max_tokens)
        return response['choices'][0]['text'].strip()

    def make_query_prompt(self, query):
        return string.Template(self.query_prompt_template).substitute(query=query)

    def make_memory_prompt(self, passage, context):
        return string.Template(self.memory_prompt_template).substitute(passage=passage, context=context)

    def post_process_query(self, query: str):
        return json.loads(query)["processed"]

    def post_process_memory(self, memory: str):
        return "\n".join(json.loads(memory)["processed"].values())


class OpenAIChatRewriteModel(OpenAIRewriteModel):
    """
    Rewrite model wrapper for OpenAI chat models such as gpt-3.5-turbo and gpt-4.
    """

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 api_key: str = None,
                 temperature: float = 0.7,
                 max_tokens: int = 512,
                 query_rewrite_prompt: str = None,
                 memory_rewrite_prompt: str = None,
                 query_rewrite_system_message: str = None,
                 memory_rewrite_system_message: str = None):
        super().__init__(model_name, api_key, temperature, max_tokens, query_rewrite_prompt, memory_rewrite_prompt)
        self.query_rewrite_system_message = query_rewrite_system_message or DEFAULT_QUERY_SYSTEM_MESSAGE
        self.memory_rewrite_system_message = memory_rewrite_system_message or DEFAULT_MEMORY_SYSTEM_MESSAGE

    def rewrite_query(self, query: str) -> str:
        prompt = self.make_query_prompt(query)
        return self.post_process_query(self.chat_completion(prompt, self.query_rewrite_system_message))

    def rewrite_memory(self, passage: str, context: str) -> str:
        prompt = self.make_memory_prompt(passage, context)
        return self.post_process_memory(self.chat_completion(prompt, self.memory_rewrite_system_message))

    def chat_completion(self, prompt, system_message):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(model=self.model_name, messages=messages, temperature=self.temperature,
                                                max_tokens=self.max_tokens)
        return response['choices'][0]['message']['content'].strip()

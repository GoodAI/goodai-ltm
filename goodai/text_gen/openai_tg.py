import threading
import openai
from typing import Optional
from goodai.text_gen.base import BaseTextGenerationModel

_openai_lock = threading.Lock()


class OpenAICompletionModel(BaseTextGenerationModel):
    def __init__(self, model_name: str, api_key: str = None, temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        with _openai_lock:
            if self.api_key:
                openai.api_key = self.api_key
            response = openai.Completion.create(model=self.model_name, prompt=prompt, temperature=self.temperature,
                                                max_tokens=self.max_tokens)
            return response['choices'][0]['text'].strip()


class OpenAIChatCompletionModel(BaseTextGenerationModel):
    def __init__(self, model_name: str, api_key: str = None, temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        with _openai_lock:
            if self.api_key:
                openai.api_key = self.api_key
            messages = [
                {"role": "user", "content": prompt}
            ]
            response = openai.ChatCompletion.create(model=self.model_name, messages=messages,
                                                    temperature=self.temperature,
                                                    max_tokens=self.max_tokens)
            return response['choices'][0]['message']['content'].strip()

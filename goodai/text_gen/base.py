from abc import ABC, abstractmethod


class BaseTextGenerationModel(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

from pyltm.embeddings import BaseTextEmbeddingModel
from pyltm.matching import BaseTextMatchingModel


@dataclass
class RetrievedMemory:
    passage: str
    distance: float
    confidence: Optional[float]
    metadata: Any


class BaseTextMemory(ABC):
    @abstractmethod
    def add_text(self, text: str, metadata: Optional[Any]):
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int, **kwargs) -> List[RetrievedMemory]:
        pass


class AutoTextMemory:
    def create(self, **kwargs) -> BaseTextMemory:
        # TODO
        pass

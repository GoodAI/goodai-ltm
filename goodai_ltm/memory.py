from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from transformers import AutoTokenizer

from goodai_ltm.embeddings import AutoTextEmbeddingModel
from goodai_ltm.memory_models.config import TextMemoryConfig
from goodai_ltm.memory_models.default import DefaultTextMemory
from goodai_ltm.memory_models.simple_vector_db import SimpleVectorDb


@dataclass
class RetrievedMemory:
    passage: str
    """
    The (expanded) passage text
    """

    distance: float
    """
    A distance metric between the retrieved passage and the query
    """

    confidence: Optional[float]
    """
    A confidence metric between 0 and 1. Not all memories support this, so it may be None
    """

    metadata: Any
    """
    Metadata associated with the retrieved text
    """


class BaseTextMemory(ABC):
    @abstractmethod
    def add_text(self, text: str, metadata: Optional[Any] = None):
        pass

    @abstractmethod
    def retrieve_multiple(self, queries: List[str], k: int, show_progress_bar: bool = False,
                          **kwargs) -> List[List[RetrievedMemory]]:
        pass

    def retrieve(self, query: str, k: int, **kwargs) -> List[RetrievedMemory]:
        multi_result = self.retrieve_multiple([query], k=k, **kwargs)
        return multi_result[0]

    @abstractmethod
    def clear(self):
        pass


class AutoTextMemory:
    @staticmethod
    def create(**kwargs) -> BaseTextMemory:
        new_kwargs = dict(kwargs)
        if 'vector_db' not in new_kwargs:
            new_kwargs['vector_db'] = SimpleVectorDb()
        if 'tokenizer' not in new_kwargs:
            new_kwargs['tokenizer'] = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
        if 'emb_model' not in new_kwargs:
            # TODO use our own
            new_kwargs['emb_model'] = AutoTextEmbeddingModel.from_pretrained('st:sentence-transformers/all-distilroberta-v1')
        if 'matching_model' not in new_kwargs:
            # TODO use our own
            new_kwargs['matching_model'] = None
        if 'device' not in new_kwargs:
            new_kwargs['device'] = torch.device('cpu')
        if 'config' not in new_kwargs:
            new_kwargs['config'] = TextMemoryConfig()
        return DefaultTextMemory(**new_kwargs)

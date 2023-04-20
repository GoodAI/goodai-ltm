import torch
from transformers import AutoTokenizer

from goodai.ltm.embedding_models.auto import AutoTextEmbeddingModel
from goodai.ltm.memory import BaseTextMemory
from goodai.ltm.memory_models.config import TextMemoryConfig
from goodai.ltm.memory_models.default import DefaultTextMemory
from goodai.ltm.memory_models.simple_vector_db import SimpleVectorDb


class AutoTextMemory:
    """
    Factory class for text memory.
    """

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

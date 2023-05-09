import enum
import sys

import torch
from typing import Union
from transformers import AutoTokenizer, PreTrainedTokenizer
from goodai.ltm.embeddings.auto import AutoTextEmbeddingModel
from goodai.ltm.embeddings.base import BaseTextEmbeddingModel
from goodai.ltm.mem.base import BaseTextMemory
from goodai.ltm.mem.config import TextMemoryConfig
from goodai.ltm.mem.default import DefaultTextMemory
from goodai.ltm.mem.mem_foundation import VectorDbType
from goodai.ltm.mem.rewrite_model import BaseRewriteModel
from goodai.ltm.reranking.base import BaseTextMatchingModel

_default_emb_model = None
_default_tokenizer = None


class MemType(enum.Enum):
    TRANSIENT_CHUNK_EMB = 0


class AutoTextMemory:
    """
    Factory class for text memory.
    """

    @staticmethod
    def create(mem_type: MemType = MemType.TRANSIENT_CHUNK_EMB,
               vector_db_type: VectorDbType = VectorDbType.SIMPLE,
               tokenizer: PreTrainedTokenizer = None,
               emb_model: BaseTextEmbeddingModel = None,
               matching_model: BaseTextMatchingModel = None,
               memory_rewrite_model: BaseRewriteModel = None,
               device: Union[torch.device, str] = None,
               config: TextMemoryConfig = None
               ) -> BaseTextMemory:
        """
        Creates a memory instance.
        :param mem_type: Reserved parameter.
        :param vector_db_type: The type of vector database. Default is VectorDbType.SIMPLE.
        :param tokenizer: A chunking tokenizer. It should be a tokenizer that preserves whitespace and casing.
        :param emb_model: The embedding model.
        :param matching_model: An optional query-passage matching model.
        :param memory_rewrite_model: The memory rewrite model.
        :param device: The Pytorch device.
        :param config: The memory configuration.
        :return: An instance of BaseTextMemory.
        """
        if tokenizer is None:
            global _default_tokenizer

            if _default_tokenizer is None:
                _default_tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
            tokenizer = _default_tokenizer
            # Suppress length warning
            tokenizer.model_max_length = sys.maxsize
        if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if emb_model is None:
            global _default_emb_model

            if _default_emb_model is None:
                _default_emb_model = AutoTextEmbeddingModel.from_pretrained('em-distilroberta-p1-01',
                                                                            device=device)
            emb_model = _default_emb_model
        if config is None:
            config = TextMemoryConfig()
        return DefaultTextMemory(vector_db_type, tokenizer, emb_model, matching_model,
                                 device=device, config=config,
                                 memory_rewrite_model=memory_rewrite_model)

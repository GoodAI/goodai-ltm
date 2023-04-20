from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import torch


class BaseTextEmbeddingModel(ABC):
    """
    Abstract base class for text embedding models.

    Text embedding models allow different embeddings for queries (retrieval embeddings) and passages
    (storage embeddings) as well as multiple retrieval and storage embeddings for a query or passage.
    """

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        :return: The number of dimensions of embedding vectors.
        """
        pass

    @abstractmethod
    def get_num_retrieval_embeddings(self) -> int:
        """
        :return: The number of retrieval embeddings produced by the model, per query.
        """
        pass

    @abstractmethod
    def get_num_storage_embeddings(self) -> int:
        """
        :return: The number of storage embeddings produced by the model, per passage.
        """
        pass

    @abstractmethod
    def encode_queries(self, queries: List[str], batch_size: int = 64, show_progress_bar: bool = False,
                       convert_to_tensor: bool = False,
                       device: Union[str, torch.device] = None) -> Union[np.ndarray, torch.Tensor]:
        pass

    @abstractmethod
    def encode_corpus(self, passages: List[str], batch_size: int = 64, show_progress_bar: bool = False,
                      convert_to_tensor: bool = False,
                      device: Union[str, torch.device] = None) -> Union[np.ndarray, torch.Tensor]:
        pass

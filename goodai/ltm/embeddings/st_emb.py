import threading

import numpy as np
import torch
from typing import List, Union

from sentence_transformers import SentenceTransformer

from goodai.helpers.torch_helper import param_count
from goodai.ltm.embeddings.base import BaseTextEmbeddingModel

_openai_lock = threading.Lock()


class SentenceTransformerEmbeddingModel(BaseTextEmbeddingModel):
    """
    Text embedding model based on Hugging Face Sentence Transformers.

    https://huggingface.co/sentence-transformers
    """

    def __init__(self, model_name: str, device: Union[torch.device, str] = None, **kwargs):
        self.st = SentenceTransformer(model_name, device=device, **kwargs)
        self.model_name = model_name

    def get_embedding_dim(self) -> int:
        return self.st.get_sentence_embedding_dimension()

    def get_num_retrieval_embeddings(self) -> int:
        return 1

    def get_num_storage_embeddings(self) -> int:
        return 1

    def get_info(self) -> str:
        return f'SentenceTransformer | Name: {self.model_name} | ' \
               f'Params: {param_count(self.st)/1e+6:.4g} million | Dimensions: {self.get_embedding_dim()}'

    def encode(self, sentences: List[str], batch_size: int = 64, show_progress_bar: bool = False,
               convert_to_tensor: bool = False,
               device: Union[str, torch.device] = None) -> Union[np.ndarray, torch.Tensor]:
        emb = self.st.encode(sentences, batch_size=batch_size, show_progress_bar=show_progress_bar,
                             convert_to_tensor=convert_to_tensor, convert_to_numpy=not convert_to_tensor,
                             device=device, normalize_embeddings=True)
        return emb[:, None, :]

    def encode_queries(self, queries: List[str], batch_size: int = 64, show_progress_bar: bool = False,
                       convert_to_tensor: bool = False,
                       device: Union[str, torch.device] = None) -> Union[np.ndarray, torch.Tensor]:
        return self.encode(queries, batch_size=batch_size, show_progress_bar=show_progress_bar,
                           convert_to_tensor=convert_to_tensor, device=device)

    def encode_corpus(self, passages: List[str], batch_size: int = 64, show_progress_bar: bool = False,
                      convert_to_tensor: bool = False,
                      device: Union[str, torch.device] = None) -> Union[np.ndarray, torch.Tensor]:
        return self.encode(passages, batch_size=batch_size, show_progress_bar=show_progress_bar,
                           convert_to_tensor=convert_to_tensor, device=device)

import threading

import numpy as np
import openai
import torch
from typing import List, Union

from tqdm import tqdm

from pyltm.embeddings import BaseTextEmbeddingModel

_openai_lock = threading.Lock()


class OpenAIEmbeddingModel(BaseTextEmbeddingModel):
    def __init__(self, api_key: str = None, model_name: str = 'text-embedding-ada-002', emb_dim: int = 1536):
        self.api_key = api_key
        self.emb_dim = emb_dim
        self.model_name = model_name

    def get_embedding_dim(self) -> int:
        return self.emb_dim

    def get_num_retrieval_embeddings(self) -> int:
        return 1

    def get_num_storage_embeddings(self) -> int:
        return 1

    def encode(self, sentences: List[str], batch_size: int = 64, show_progress_bar: bool = False,
               convert_to_tensor: bool = False,
               device: Union[str, torch.device] = None) -> Union[np.ndarray, torch.Tensor]:
        with _openai_lock:
            openai.api_key = self.api_key
            rng = range(0, len(sentences), batch_size)
            if show_progress_bar:
                rng = tqdm(rng, desc='Embeddings', unit='batch')
            all_emb_vectors = []
            for b0 in rng:
                b_queries = sentences[b0:b0 + batch_size]
                response = openai.Embedding.create(input=b_queries, model=self.model_name)
                data_array = response['data']
                emb_vectors = [entry['embedding'] for entry in data_array]
                all_emb_vectors.extend(emb_vectors)
        if convert_to_tensor:
            return torch.as_tensor(emb_vectors, dtype=torch.float, device=device).unsqueeze(1)
        else:
            return np.array(all_emb_vectors)[:, None, :]

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

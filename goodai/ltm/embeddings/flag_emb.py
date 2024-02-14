from typing import List, Union

import numpy as np
import torch
from FlagEmbedding import FlagModel

from goodai.ltm.embeddings.base import BaseTextEmbeddingModel


class FlagEmbeddingModel(BaseTextEmbeddingModel):
    def __init__(self, model_name: str, **kwargs):
        self.fm = FlagModel(model_name, **kwargs)
        self.model_name = model_name

    def get_embedding_dim(self) -> int:
        return self.fm.model.config.hidden_size

    def get_num_retrieval_embeddings(self) -> int:
        return 1

    def get_num_storage_embeddings(self) -> int:
        return 1

    def get_info(self) -> str:
        return f"FlagModel:{self.model_name}"

    @staticmethod
    def _result(tensor: np.ndarray, convert_to_tensor: bool,
                device: Union[str, torch.device]) -> Union[np.ndarray, torch.Tensor]:
        if len(tensor.shape) == 2:
            tensor = tensor[:, None, :]
        if convert_to_tensor:
            return torch.from_numpy(tensor).to(device)
        else:
            return tensor

    def encode_queries(self, queries: List[str], batch_size: int = 64,
                       show_progress_bar: bool = False, convert_to_tensor: bool = False,
                       device: Union[str, torch.device] = None) -> Union[np.ndarray, torch.Tensor]:
        result = self.fm.encode_queries(queries=queries, batch_size=batch_size,
                                        convert_to_numpy=True)
        return self._result(result, convert_to_tensor=convert_to_tensor, device=device)

    def encode_corpus(self, passages: List[str], batch_size: int = 64,
                      show_progress_bar: bool = False, convert_to_tensor: bool = False,
                      device: Union[str, torch.device] = None) -> Union[np.ndarray, torch.Tensor]:
        result = self.fm.encode_corpus(passages, batch_size=batch_size,
                                       convert_to_numpy=True)
        return self._result(result, convert_to_tensor=convert_to_tensor, device=device)

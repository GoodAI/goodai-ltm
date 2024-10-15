import torch
import numpy as np
from multiprocessing import Queue
from goodai.ltm.embeddings.base import BaseTextEmbeddingModel


class RemoteEmbeddingModel(BaseTextEmbeddingModel):
    """
    Supports embedding models running on different processes or machines.
    Jobs queue must support remote calls of methods 'get_embedding_dim', 'get_info', and
    'encode' --> {"method": "method_name", "args": [...], "kwargs": {...}}
    The method 'encode' must follow the SentenceTransformer specification.
    """

    def __init__(self, jobs_queue: Queue = None, results_queue: Queue = None):
        self._jobs_queue = jobs_queue or Queue()
        self._results_queue = results_queue or Queue()

    def _remote_call(self, method: str, *args, **kwargs):
        self._jobs_queue.put(dict(method=method, args=args, kwargs=kwargs))
        return self._results_queue.get()

    def get_embedding_dim(self) -> int:
        return self._remote_call("get_embedding_dim")

    def get_num_retrieval_embeddings(self) -> int:
        return 1

    def get_num_storage_embeddings(self) -> int:
        return 1

    def get_info(self) -> str:
        return self._remote_call("get_info")

    def encode(self, sentences: list[str], batch_size: int = 64, convert_to_tensor: bool = False,
               device: str | torch.device = None) -> np.ndarray | torch.Tensor:
        emb = self._remote_call("encode", sentences, batch_size=batch_size,
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
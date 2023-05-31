from abc import abstractmethod
from typing import List, Tuple, Union, Callable

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from goodai.helpers.tokenizer_helper import get_pad_token_id, get_model_inputs
from goodai.ltm.embeddings.base import BaseTextEmbeddingModel


class TrainableEmbeddingModel(BaseTextEmbeddingModel, nn.Module):
    """
    Abstract base class for locally trainable text embeddings.

    Trainable embedding models support multiple retrieval and storage embeddings for a query or passage.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(BaseTextEmbeddingModel, self).__init__()
        super(nn.Module, self).__init__()
        self.dummy = nn.Parameter()
        self.tokenizer = tokenizer
        self.pad_token_id = get_pad_token_id(tokenizer)

    def get_device(self):
        return self.dummy.device

    @abstractmethod
    def get_storage_emb(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Returns an embedding of shape (batch_size, num_keys, emb_size,)
        """
        pass

    @abstractmethod
    def get_retrieval_emb(self, input_ids: torch.Tensor, token_lengths: torch.Tensor,
                          attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Returns an embedding of shape (batch_size, emb_size,)
        """
        pass

    @abstractmethod
    def get_emb(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns storage and retrieval embeddings.
        """
        pass

    @abstractmethod
    def get_lm_parameters(self):
        """
        :return: Fine-tuning parameters belonging to a pretrained language model.
        """
        pass

    @abstractmethod
    def get_added_parameters(self):
        """
        :return: New parameters that don't belong to the pretrained language model.
        """
        pass

    def get_storage_emb_for_ids(self, input_ids: List[List[int]]):
        device = self.get_device()
        pad_token_id = self.pad_token_id
        m_inputs = get_model_inputs(input_ids, pad_id=pad_token_id, device=device)
        return self.get_storage_emb(**m_inputs)

    def get_retrieval_emb_for_ids(self, input_ids: List[List[int]]):
        device = self.get_device()
        pad_token_id = self.pad_token_id
        m_inputs = get_model_inputs(input_ids, pad_id=pad_token_id, device=device, return_token_lengths=True,
                                    tokenizer=self.tokenizer)
        return self.get_retrieval_emb(**m_inputs)

    def _get_token_ids(self, texts: List[str]) -> List[List[int]]:
        tok = self.tokenizer
        return [tok.encode(text, add_special_tokens=False) for text in texts]

    def encode_in_batches(self, enc_fn: Callable, sentences: List[str],
                          add_special_tokens: bool = True,
                          batch_size: int = 50,
                          show_progress_bar: bool = False,
                          convert_to_tensor: bool = False,
                          return_token_lengths: bool = False,
                          detach_tensors: bool = True,
                          device: Union[str, torch.device] = None):
        device = device if device else self.get_device()
        t = self.tokenizer
        s_count = len(sentences)
        rng = range(0, s_count, batch_size)
        if show_progress_bar:
            rng = tqdm(rng, desc='Embeddings', unit='batch')
        keys_list = []
        for b0 in rng:
            b_sentences = sentences[b0:b0 + batch_size]
            input_ids_list = [t.encode(s, add_special_tokens=add_special_tokens) for s in b_sentences]
            model_inputs = get_model_inputs(input_ids_list, self.pad_token_id,
                                            return_token_lengths=return_token_lengths,
                                            tokenizer=self.tokenizer,
                                            device=device)
            keys = enc_fn(**model_inputs)
            if detach_tensors or not convert_to_tensor:
                keys = keys.detach()
            keys_list.append(keys)
        result = torch.cat(keys_list)
        if convert_to_tensor:
            return result
        else:
            return result.detach().cpu().numpy()

    def encode_queries(self, queries: List[str], batch_size: int = 64, show_progress_bar: bool = False,
                       convert_to_tensor: bool = False, detach_tensors: bool = True,
                       device: Union[str, torch.device] = None) -> Union[np.ndarray, torch.Tensor]:
        return self.encode_in_batches(self.get_retrieval_emb, queries, batch_size=batch_size,
                                      show_progress_bar=show_progress_bar,
                                      convert_to_tensor=convert_to_tensor,
                                      detach_tensors=detach_tensors,
                                      return_token_lengths=True,
                                      device=device)

    def encode_corpus(self, passages: List[str], batch_size: int = 64, show_progress_bar: bool = False,
                      convert_to_tensor: bool = False, detach_tensors: bool = True,
                      device: Union[str, torch.device] = None) -> Union[np.ndarray, torch.Tensor]:
        return self.encode_in_batches(self.get_storage_emb, passages, batch_size=batch_size,
                                      show_progress_bar=show_progress_bar,
                                      convert_to_tensor=convert_to_tensor,
                                      detach_tensors=detach_tensors,
                                      return_token_lengths=False,
                                      device=device)

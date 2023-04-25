from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseQueryPassageProbModel(ABC):
    """
    Abstract base class for query-passage probability models.
    """

    @abstractmethod
    def forward(self, query_input_ids: torch.Tensor, query_attention_mask: torch.Tensor,
                query_token_lengths: torch.Tensor,
                passage_input_ids: torch.Tensor, passage_attention_mask: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_lm_parameters(self):
        pass

    @abstractmethod
    def get_added_parameters(self):
        pass

    @abstractmethod
    def train(self, mode: bool = True):
        pass

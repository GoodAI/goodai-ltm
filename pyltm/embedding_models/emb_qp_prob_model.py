import itertools
import math

import torch
from torch import nn

from pyltm.embedding_models.contrast_classifier import ContrastClassifier
from pyltm.embedding_models.trainable import TrainableEmbeddingModel
from pyltm.matching_models.prob_model import BaseQueryPassageProbModel


class EmbeddingQueryPassageProbModel(nn.Module, BaseQueryPassageProbModel):
    def __init__(self, emb_model: TrainableEmbeddingModel):
        super(EmbeddingQueryPassageProbModel, self).__init__()
        self.emb_model = emb_model
        scale = math.sqrt(emb_model.get_embedding_dim())
        self.classifier = ContrastClassifier(scale)

    def forward(self, query_input_ids: torch.Tensor, query_attention_mask: torch.Tensor,
                query_token_lengths: torch.Tensor,
                passage_input_ids: torch.Tensor, passage_attention_mask: torch.Tensor) -> torch.Tensor:
        rk = self.emb_model.get_retrieval_key(query_input_ids, token_lengths=query_token_lengths,
                                              attention_mask=query_attention_mask)
        sk = self.emb_model.get_storage_key(passage_input_ids, passage_attention_mask)
        return self.classifier(rk, sk)

    def get_lm_parameters(self):
        return self.emb_model.get_lm_parameters()

    def get_added_parameters(self):
        km_added_parameters = self.emb_model.get_added_parameters()
        c_parameters = self.classifier.parameters()
        return itertools.chain(km_added_parameters, c_parameters)


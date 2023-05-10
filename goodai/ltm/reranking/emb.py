import gc
from typing import List, Tuple

import torch

from goodai.helpers.tokenizer_helper import get_model_inputs
from goodai.ltm.embeddings.base import BaseTextEmbeddingModel
from goodai.ltm.embeddings.trainable import TrainableEmbeddingModel
from goodai.ltm.reranking.base import BaseTextMatchingModel
from goodai.modules.loss import EmbCrossProbLossModel


class EmbeddingBasedMatchingModel(BaseTextMatchingModel):
    def __init__(self, emb_model: BaseTextEmbeddingModel, dist_param: float):
        super().__init__()
        self.dist_param = dist_param
        self.emb_model = emb_model

    def predict(self, sentences: List[Tuple[str, str]], batch_size: int = 32,
                show_progress_bar: bool = False, add_special_tokens: bool = True) -> List[float]:
        queries, passages = zip(*sentences)
        rk = self.emb_model.encode_queries(queries, batch_size=batch_size, show_progress_bar=show_progress_bar,
                                           convert_to_tensor=True)
        rk = rk.detach()
        gc.collect()
        sk = self.emb_model.encode_corpus(passages, batch_size=batch_size, show_progress_bar=show_progress_bar,
                                          convert_to_tensor=True)
        sk = sk.detach()
        # rk: (batch_size, num_r_emb, emb_dim,)
        # sk: (batch_size, num_s_emb, emb_dim,)
        distances = torch.cdist(rk, sk)
        # distances: (batch_size, num_r_emb, num_s_emb,)
        batch_size = distances.size(0)
        distances = distances.view(batch_size, -1)
        min_distances = torch.amin(distances, dim=1, keepdim=True)
        prob_t = EmbCrossProbLossModel.get_prob(min_distances, self.dist_param)
        return prob_t.squeeze(1).tolist()

    def get_info(self):
        return f'{self.__class__}: EmbModel {self.emb_model.get_info()}'

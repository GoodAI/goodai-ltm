import torch
from typing import List, Tuple
from tqdm import tqdm
from goodai.ltm.embeddings.base import BaseTextEmbeddingModel
from goodai.ltm.reranking.base import BaseTextMatchingModel
from goodai.modules.loss import EmbCrossProbLossModel


class EmbeddingBasedMatchingModel(BaseTextMatchingModel):
    def __init__(self, emb_model: BaseTextEmbeddingModel, dist_param: float):
        super().__init__()
        self.dist_param = dist_param
        self.emb_model = emb_model

    def predict(self, sentences: List[Tuple[str, str]], batch_size: int = 32,
                show_progress_bar: bool = False, add_special_tokens: bool = True) -> List[float]:
        with torch.no_grad():
            rng = range(0, len(sentences), batch_size)
            if show_progress_bar:
                rng = tqdm(rng, desc='Encoding QPs', unit='batch')
            result: List[float] = []
            for b0 in rng:
                b_sentences = sentences[b0:b0 + batch_size]
                queries, passages = zip(*b_sentences)
                rk = self.emb_model.encode_queries(queries, batch_size=batch_size, show_progress_bar=False,
                                                   convert_to_tensor=True)
                rk = rk.detach()
                sk = self.emb_model.encode_corpus(passages, batch_size=batch_size, show_progress_bar=False,
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
                partial_result = prob_t.squeeze(1).tolist()
                result.extend(partial_result)
            return result

    def get_info(self):
        return f'{self.__class__}: EmbModel {self.emb_model.get_info()}'

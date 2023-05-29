import pickle
from typing import Union
from weakref import WeakValueDictionary

import torch

from goodai.helpers.file_helper import unpickle_downloaded_url
from goodai.ltm.embeddings.auto import AutoTextEmbeddingModel
from goodai.ltm.reranking.base import BaseTextMatchingModel
from goodai.ltm.reranking.default import DefaultRerankingCrossEncoder
from goodai.ltm.reranking.emb import EmbeddingBasedMatchingModel
from goodai.ltm.reranking.st_ce import SentenceTransformerTextMatchingModel

_default_dist_param = 0.75
_models_base = 'https://github.com/GoodAI/goodai-ltm-artifacts/releases/download/models/goodai-ltm-qpm-model'
_pretrained_map = {
    'qpm-distilroberta-01': f'{_models_base}-1145'
}


class AutoTextMatchingModel:
    """
    Factory class for text matching models.
    """

    @staticmethod
    def shared_pretrained(name: str, device: Union[str, torch.device] = None):
        if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        key = f'{name}|{device}'
        model = AutoTextMatchingModel.model_cache.get(key)
        if model is None:
            model = AutoTextMatchingModel.from_pretrained(name, device)
            AutoTextMatchingModel.model_cache[key] = model
        return model

    @staticmethod
    def from_pretrained(name: str, device: Union[torch.device, str] = None) -> BaseTextMatchingModel:
        if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        name = name.strip()
        colon_idx = name.find(':')
        if colon_idx == -1:
            url = _pretrained_map.get(name)
            if url is None:
                raise ValueError(f'GoodAI model not found: {name}')
            model_dict = unpickle_downloaded_url(url)
            model: DefaultRerankingCrossEncoder = model_dict['qpmm']
            model.to(device)
            model.zero_grad(set_to_none=True)
            model.eval()
            return model
        model_type = name[:colon_idx]
        model_name = name[colon_idx + 1:]
        if model_type == 'st':
            return SentenceTransformerTextMatchingModel(model_name)
        elif model_type == 'em':
            emb_model = AutoTextEmbeddingModel.shared_pretrained(model_name, device=device)
            return EmbeddingBasedMatchingModel(emb_model, _default_dist_param)
        else:
            raise ValueError(f'Unknown model type: {model_type}')


AutoTextMatchingModel.model_cache = WeakValueDictionary()
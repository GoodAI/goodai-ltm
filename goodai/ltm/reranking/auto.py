import pickle
from typing import Union
import torch

from goodai.helpers.file_helper import open_url_as_file
from goodai.ltm.reranking.base import BaseTextMatchingModel
from goodai.ltm.reranking.default import DefaultRerankingCrossEncoder
from goodai.ltm.reranking.st_ce import SentenceTransformerTextMatchingModel

_models_base = 'https://github.com/GoodAI/goodai-ltm-artifacts/releases/download/models/goodai-ltm-qpm-model'


_pretrained_map = {
    'qpm-distilroberta-01': f'{_models_base}-1145'
}


class AutoTextMatchingModel:
    """
    Factory class for text matching models.
    """

    @staticmethod
    def from_pretrained(name: str, device: Union[torch.device, str] = None) -> BaseTextMatchingModel:
        if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        name = name.strip()
        colon_idx = name.find(':')
        if colon_idx == -1:
            url = _pretrained_map.get(name)
            if url is None:
                raise ValueError(f'Model not found: {name}')
            with open_url_as_file(url) as fd:
                model_dict = pickle.load(fd)
                model: DefaultRerankingCrossEncoder = model_dict['qpmm']
                model.to(device)
                model.zero_grad(set_to_none=True)
                model.eval()
                return model
        model_type = name[:colon_idx]
        model_name = name[colon_idx + 1:]
        if model_type == 'st':
            return SentenceTransformerTextMatchingModel(model_name)
        else:
            raise ValueError(f'Unknown model type: {model_type}')

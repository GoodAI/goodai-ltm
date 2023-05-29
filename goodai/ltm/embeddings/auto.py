import pickle
from typing import Union
from weakref import WeakValueDictionary

import torch

from goodai.helpers.file_helper import open_url_as_file, unpickle_downloaded_url
from goodai.ltm.embeddings.openai_emb import OpenAIEmbeddingModel
from goodai.ltm.embeddings.st_emb import SentenceTransformerEmbeddingModel
from goodai.ltm.embeddings.trainable import TrainableEmbeddingModel
from goodai.ltm.embeddings.base import BaseTextEmbeddingModel

_models_base = 'https://github.com/GoodAI/goodai-ltm-artifacts/releases/download/models/goodai-ltm-emb-model'

_pretrained_map = {
    'em-distilroberta-p1-01': f'{_models_base}-p1-1118',
    'em-distilroberta-p3-01': f'{_models_base}-p3-1116',
    'em-MiniLM-p3-01': f'{_models_base}-p3-1117',
    'em-MiniLM-p1-01': f'{_models_base}-p1-1151',
    'em-distilroberta-p5-01': f'{_models_base}-p5-1150',
}


class AutoTextEmbeddingModel:
    """
    Factory class for text embedding models.
    """

    @staticmethod
    def shared_pretrained(name: str, device: Union[str, torch.device] = None):
        if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        key = f'{name}|{device}'
        model = AutoTextEmbeddingModel.model_cache.get(key)
        if model is None:
            model = AutoTextEmbeddingModel.from_pretrained(name, device)
            AutoTextEmbeddingModel.model_cache[key] = model
        return model

    @staticmethod
    def from_pretrained(name: str, device: Union[str, torch.device] = None, **kwargs) -> BaseTextEmbeddingModel:
        """
        Makes a pretrained embedding model from a descriptor (name).

        The name has the format <model_type>:<model_name>, for example "st:sentence-transformers/all-distilroberta-v1",
        where model_type is 'st' for Hugging Face Sentence Transformers or 'openai' for Open AI text embeddings.

        :param name: Name in the format <model_type>:<model_name>
        :param device: The Pytorch device for the model, if applicable
        :return: The embedding model
        """

        if device is None:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        name = name.strip()
        colon_idx = name.find(':')
        if colon_idx == -1:
            url = _pretrained_map.get(name)
            if url is None:
                raise ValueError(f'GoodAI model not found: {name}')
            model_dict = unpickle_downloaded_url(url)
            model: TrainableEmbeddingModel = model_dict['emb_model']
            model.to(device)
            model.zero_grad(set_to_none=True)
            model.eval()
            return model
        model_type = name[:colon_idx]
        model_name = name[colon_idx + 1:]
        if model_type == 'st':
            return SentenceTransformerEmbeddingModel(model_name, device=device, **kwargs)
        elif model_type == 'openai':
            return OpenAIEmbeddingModel(model_name, device=device, **kwargs)
        else:
            raise ValueError(f'Unknown model type: {model_type}')


AutoTextEmbeddingModel.model_cache = WeakValueDictionary()

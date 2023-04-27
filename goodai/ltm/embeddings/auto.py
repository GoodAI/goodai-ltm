import pickle
from typing import Union

import torch

from goodai.helpers.file_helper import open_url_as_file
from goodai.ltm.embeddings.openai_emb import OpenAIEmbeddingModel
from goodai.ltm.embeddings.st_emb import SentenceTransformerEmbeddingModel
from goodai.ltm.embeddings.trainable import TrainableEmbeddingModel
from goodai.ltm.embeddings.base import BaseTextEmbeddingModel

_models_base = 'https://github.com/GoodAI/goodai-ltm-artifacts/releases/download/models'

_pretrained_map = {
    'p2-qa-mpnet': f'{_models_base}/goodai-ltm-emb-model-p2-1063',
    'p3-distilroberta': f'{_models_base}/goodai-ltm-emb-model-p3-1062',
    'p3-reserved-01': f'{_models_base}/goodai-ltm-emb-model-p3-1064',
    'p3-reserved-02': f'{_models_base}/goodai-ltm-emb-model-p3-1067',
    'p1-reserved-03': f'{_models_base}/goodai-ltm-emb-model-p1-1068',
    'p1-reserved-04': f'{_models_base}/goodai-ltm-emb-model-p1-1069',
}


class AutoTextEmbeddingModel:
    """
    Factory class for text embedding models.
    """

    @staticmethod
    def from_pretrained(name: str, device: Union[str, torch.device] = None) -> BaseTextEmbeddingModel:
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
                raise ValueError(f'Model not found: {name}')
            with open_url_as_file(url) as fd:
                model_dict = pickle.load(fd)
                model: TrainableEmbeddingModel = model_dict['emb_model']
                model.to(device)
                model.zero_grad(set_to_none=True)
                model.eval()
                return model
        model_type = name[:colon_idx]
        model_name = name[colon_idx + 1:]
        if model_type == 'st':
            return SentenceTransformerEmbeddingModel(model_name, device=device)
        elif model_type == 'openai':
            return OpenAIEmbeddingModel(model_name)
        else:
            raise ValueError(f'Unknown model type: {model_type}')

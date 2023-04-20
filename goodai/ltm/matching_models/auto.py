from goodai.ltm.matching import BaseTextMatchingModel
from goodai.ltm.matching_models.st_ce import SentenceTransformerTextMatchingModel


class AutoTextMatchingModel:
    """
    Factory class for text matching models.
    """

    @staticmethod
    def from_pretrained(name: str) -> BaseTextMatchingModel:
        name = name.strip()
        colon_idx = name.find(':')
        if colon_idx == -1:
            raise ValueError(f'Model not available: {name}')
        model_type = name[:colon_idx]
        model_name = name[colon_idx + 1:]
        if model_type == 'st':
            return SentenceTransformerTextMatchingModel(model_name)
        else:
            # TODO pretrained models
            raise ValueError(f'Unknown model type: {model_type}')

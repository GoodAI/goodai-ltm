from goodai.ltm.embedding_models.openai_emb import OpenAIEmbeddingModel
from goodai.ltm.embedding_models.st_emb import SentenceTransformerEmbeddingModel
from goodai.ltm.embeddings import BaseTextEmbeddingModel


class AutoTextEmbeddingModel:
    """
    Factory class for text embedding models.
    """

    @staticmethod
    def from_pretrained(name: str) -> BaseTextEmbeddingModel:
        """
        Makes a pretrained embedding model from a descriptor (name).

        The name has the format <model_type>:<model_name>, for example "st:sentence-transformers/all-distilroberta-v1",
        where model_type is 'st' for Hugging Face Sentence Transformers or 'openai' for Open AI text embeddings.

        :param name: Name in the format <model_type>:<model_name>
        :return: The embedding model
        """

        name = name.strip()
        colon_idx = name.find(':')
        if colon_idx == -1:
            raise ValueError(f'Model not available: {name}')
        model_type = name[:colon_idx]
        model_name = name[colon_idx + 1:]
        if model_type == 'st':
            return SentenceTransformerEmbeddingModel(model_name)
        elif model_type == 'openai':
            return OpenAIEmbeddingModel(model_name)
        else:
            raise ValueError(f'Unknown model type: {model_type}')

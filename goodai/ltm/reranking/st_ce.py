from typing import List, Tuple

from sentence_transformers import CrossEncoder

from goodai.ltm.reranking.base import BaseTextMatchingModel


class SentenceTransformerTextMatchingModel(BaseTextMatchingModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.ce = CrossEncoder(model_name)

    def predict(self, sentences: List[Tuple[str, str]], batch_size: int = 32,
                show_progress_bar: bool = False) -> List[float]:
        results = self.ce.predict(sentences, batch_size=batch_size, show_progress_bar=show_progress_bar,
                                  convert_to_tensor=True, convert_to_numpy=False)
        return results.tolist()

    def get_info(self):
        return f'SentenceTransformer cross-encoder {self.model_name}'

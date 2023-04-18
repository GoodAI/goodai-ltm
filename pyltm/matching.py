from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseTextMatchingModel(ABC):
    def get_match_confidence(self, query: str, passages: List[str], batch_size: int = 32,
                             show_progress_bar: bool = False) -> List[float]:
        """
        :param query: A string representing a query.
        :param passages: A list of strings representing passages that should be matched with the provided query.
        :param show_progress_bar: Whether a progress bar should be shown.
        :param batch_size: The inference batch size.
        :return: A list of probabilities or confidence scores (0 to 1) quantifying query-passage matches.
        """
        sentences = [(query, p) for p in passages]
        return self.predict(sentences, batch_size=batch_size, show_progress_bar=show_progress_bar)

    @abstractmethod
    def predict(self, sentences: List[Tuple[str, str]], batch_size: int = 32,
                show_progress_bar: bool = False) -> List[float]:
        pass


class AutoTextMatchingModel:
    def from_pretrained(self, name: str) -> BaseTextMatchingModel:
        # TODO
        pass

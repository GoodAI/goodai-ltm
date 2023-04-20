from abc import ABC, abstractmethod
from typing import List
from goodai.ltm.data.query_passage.example import QueryPassageExample


class BaseQueryPassageDataSource(ABC):
    """
    Abstract interface for data sources for training and evaluation of query-passage matching models.
    """

    def __init__(self):
        pass

    @abstractmethod
    def sample_items(self, count: int) -> List[QueryPassageExample]:
        pass

from abc import ABC, abstractmethod
from typing import List
from pyltm.data.query_passage.example import QueryPassageExample


class BaseQueryPassageDataSource(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample_items(self, count: int) -> List[QueryPassageExample]:
        pass

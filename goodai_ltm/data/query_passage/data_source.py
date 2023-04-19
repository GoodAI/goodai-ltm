from abc import ABC, abstractmethod
from typing import List
from goodai_ltm.data.query_passage.example import QueryPassageExample


class BaseQueryPassageDataSource(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample_items(self, count: int) -> List[QueryPassageExample]:
        pass

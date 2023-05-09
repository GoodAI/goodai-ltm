import io
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class RetrievedMemory:
    passage: str
    """
    The (expanded) passage text
    """

    distance: float
    """
    A distance metric between the retrieved passage and the query
    """

    confidence: Optional[float]
    """
    A confidence metric between 0 and 1. Not all memories support this, so it may be None
    """

    metadata: Optional[dict]
    """
    Metadata associated with the retrieved text
    """


class BaseTextMemory(ABC):
    """
    Abstract base class for text memories.
    """

    @abstractmethod
    def add_text(self, text: str, metadata: Optional[dict] = None, rewrite: bool = False,
                 rewrite_context: Optional[str] = None, show_progress_bar: bool = False):
        """
        Adds text to the memory.
        :param show_progress_bar: Whether a progress bar should be shown
        :param text: The string that is appended to the memory
        :param metadata: An optional dictionary with metadata
        :param rewrite: Whether the text should be rewritten by an LLM
        :param rewrite_context: The context provided to the LLM for rewriting the text
        """
        pass

    @abstractmethod
    def retrieve_multiple(self, queries: List[str], k: int, rewrite: bool = False, show_progress_bar: bool = False,
                          max_query_length: Optional[int] = 40,
                          **kwargs) -> List[List[RetrievedMemory]]:
        pass

    def retrieve(self, query: str, k: int, rewrite: bool = False,
                 max_query_length: Optional[int] = 40, **kwargs) -> List[RetrievedMemory]:
        """
        Performs a memory search and retrieves relevant passages.
        :param query: The query used to search for memories
        :param k: The number of requested memories
        :param rewrite: Whether the query should be rewritten by an LLM
        :param max_query_length: The maximum number of tokens from the query that should be used
                                 to perform lookups. The left side of the query is truncated accordingly.
        :param kwargs: Additional argument passed to underlying models
        :return: A list of RetrievedMemory instances.
        """
        multi_result = self.retrieve_multiple([query], k=k, rewrite=rewrite,
                                              max_query_length=max_query_length, **kwargs)
        return multi_result[0]

    @abstractmethod
    def clear(self):
        """
        Clears all content in the memory.
        """
        pass

    @abstractmethod
    def dump(self, stream: io.TextIOBase = sys.stdout):
        """
        Performs a diagnostic dump of the contents of the memory
        :param stream: This IO stream is where text is dumped
        """
        pass


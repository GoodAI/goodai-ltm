import io
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RetrievedMemory:
    passage: str
    """
    The text of the retrieved passages, or expanded chunk.
    """

    timestamp: float
    """
    The timestamp of the memory (seconds since Epoch).
    """

    distance: float
    """
    A distance metric between the retrieved passage and the query.
    """

    relevance: float
    """
    A metric between 0 and 1 that indicates how relevant to the query 
    the retrieved memory is. It may be derived from the distance or 
    the confidence, if available.
    """

    confidence: Optional[float] = None
    """
    A confidence metric between 0 and 1. Not all memory implementations 
    support this, so it may be None.
    """

    importance: Optional[float] = None
    """
    An importance score assigned to the retrieved memory by the importance model.
    If an importance model is not available, this property will be None.
    """

    metadata: Optional[dict] = None
    """
    Metadata associated with the retrieved text.
    """


class BaseReranker(ABC):
    """
    Abstract base class for custom rerankers.
    """
    @abstractmethod
    def rerank(self, r_memories: List[RetrievedMemory], mem: 'BaseTextMemory') -> List[RetrievedMemory]:
        pass


class BaseImportanceModel(ABC):
    """
    Abstract base class for models that set the importance property of stored memories.
    """

    @abstractmethod
    def get_importance(self, mem_text: str):
        pass


class BaseTextMemory(ABC):
    """
    Abstract base class for text memories.
    """

    @abstractmethod
    def add_text(self, text: str, metadata: Optional[dict] = None, rewrite: bool = False,
                 rewrite_context: Optional[str] = None, show_progress_bar: bool = False,
                 timestamp: Optional[float] = None):
        """
        Adds text to the memory.
        :param show_progress_bar: Whether a progress bar should be shown
        :param text: The string that is appended to the memory
        :param metadata: An optional dictionary with metadata
        :param rewrite: Whether the text should be rewritten by an LLM
        :param rewrite_context: The context provided to the LLM for rewriting the text
        :param timestamp: A custom timestamp for the memory to use instead of time.time()
        """
        pass

    @abstractmethod
    def add_separator(self):
        """
        Adds a section separator. The last chunk in the memory queue will be filled with padding.
        Chunk expansion cannot go across section boundaries.
        """
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """
        :return: Whether the memory contains no tokens
        """
        pass

    @abstractmethod
    def retrieve_multiple(self, queries: List[str], k: int, rewrite: bool = False, show_progress_bar: bool = False,
                          **kwargs) -> List[List[RetrievedMemory]]:
        pass

    def retrieve(self, query: str, k: int, rewrite: bool = False, **kwargs) -> List[RetrievedMemory]:
        """
        Performs a memory search and retrieves relevant passages.
        :param query: The query used to search for memories
        :param k: The number of requested memories
        :param rewrite: Whether the query should be rewritten by an LLM
        :param kwargs: Additional argument passed to underlying models
        :return: A list of RetrievedMemory instances.
        """
        multi_result = self.retrieve_multiple([query], k=k, rewrite=rewrite,
                                              **kwargs)
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

    @abstractmethod
    def has_importance_model(self) -> bool:
        pass

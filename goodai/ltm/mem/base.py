import io
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from goodai.ltm.mem.chunk import TextKeyType
from goodai.ltm.mem.chunk_queue import PassageInfo


@dataclass
class RetrievedMemory:
    passage: str
    """
    The text of the retrieved passage, or expanded chunk.
    """

    passage_info: PassageInfo
    """
    Additional information about the passage location and tokens.
    """

    timestamp: float
    """
    The timestamp of the memory. By default, this is seconds since Epoch. The user
    may provide custom timestamps when adding text to the memory object.
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

    textKeys: List[TextKeyType]
    """
    Text keys associated with the matching chunk. These are the keys
    returned when add_text() is called.
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

    @staticmethod
    def has_overlap(included_indexes: set[int], p_from: int, p_to: int,
                    overlap_threshold: float):
        if p_from not in included_indexes and (p_to - 1) not in included_indexes:
            return False
        if p_to <= p_from:
            return False
        intersection = included_indexes.intersection(range(p_from, p_to))
        len_inter = len(intersection)
        if len_inter <= 0:
            return False
        overlap_fraction = len_inter / (p_to - p_from)
        return overlap_fraction >= overlap_threshold

    @classmethod
    def remove_overlaps(cls, r_memories: list['RetrievedMemory'],
                        overlap_threshold: float) -> list['RetrievedMemory']:
        result = []
        included_indexes: set[int] = set()
        for item in r_memories:
            p_info = item.passage_info
            if p_info is None:
                continue
            p_from = p_info.fromIndex
            p_to = p_info.toIndex
            if cls.has_overlap(included_indexes, p_from, p_to, overlap_threshold):
                continue
            result.append(item)
            included_indexes.update(range(p_from, p_to))
        return result


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
                 timestamp: Optional[float] = None) -> TextKeyType:
        """
        Adds text to the memory.
        :param show_progress_bar: Whether a progress bar should be shown
        :param text: The string that is appended to the memory
        :param metadata: An optional dictionary with metadata
        :param rewrite: Whether the text should be rewritten by an LLM
        :param rewrite_context: The context provided to the LLM for rewriting the text
        :param timestamp: A custom timestamp for the memory to use instead of time.time()
        :return: A unique key associated with the text that was added to the memory.
        """
        pass

    @abstractmethod
    def replace_text(self, text_key: TextKeyType, text: str, metadata: Optional[dict] = None,
                     rewrite: bool = False, rewrite_context: Optional[str] = None,
                     show_progress_bar: bool = False, timestamp: Optional[float] = None) -> TextKeyType:
        """
        Replaces text stored in the memory.
        :param text_key: A key previously returned by the add_text() method.
        :param text: The new text.
        :param metadata: Any metadata associated with the new text.
        :param rewrite: Whether the text should be rewritten by an LLM.
        :param rewrite_context: The context passed to the LLM to inform the rewrite.
        :param show_progress_bar: Whether a progress bar should be shown.
        :param timestamp: The timestamp associated with the new text.
        :return: The text key.
        """
        pass

    @abstractmethod
    def delete_text(self, text_key: TextKeyType, show_progress_bar: bool = False) -> TextKeyType:
        """
        Deletes text stored in the meory.
        :param text_key: A key previously returned by the add_text() method.
        :param show_progress_bar: Whether a progress bar should be shown.
        :return: The text key.
        """
        pass

    @abstractmethod
    def get_text(self, text_key: TextKeyType) -> Optional[str]:
        """
        :param text_key: A key previously returned by the add_text() method.
        :return: The text associated with the provided key. If the key does not exist, the returned value is None.
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

    @abstractmethod
    def state_as_text(self) -> str:
        """
        Returns the contents of the memory as text.
        Use set_state() to replace contents.
        """
        pass

    @abstractmethod
    def set_state(self, state: str):
        """
        Replaces the contents of the memory with state previously
        obtained using state_as_text().
        """
        pass

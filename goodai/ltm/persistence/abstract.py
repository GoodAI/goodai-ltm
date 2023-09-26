from abc import ABC, abstractmethod
from pathlib import Path

from goodai.ltm.mem.default import DefaultTextMemory


class MemoryPersistence(ABC):
    """
    An interface that defines various strategies for persisting the memory to disk.
    This abstract base class ensures that any derived class implements both the
    save and load methods for memory persistence.
    """

    @abstractmethod
    def save(self, memory: DefaultTextMemory, directory: Path) -> None:
        """
        Save the provided memory to the specified directory.

        :param memory: The memory instance to be saved.
        :type memory: DefaultTextMemory
        :param directory: The directory where the memory should be saved.
        :type directory: Path
        """
        pass

    @abstractmethod
    def load(self, directory: Path, **kwargs) -> DefaultTextMemory:
        """
        Load the memory from the specified directory.

        :param directory: The directory from which the memory should be loaded.
        :type directory: Path
        :param kwargs: Additional keyword arguments passed to AutoTextMemory.create.
        :return: The loaded memory instance.
        :rtype: DefaultTextMemory
        """
        pass

    @abstractmethod
    def exists(self, directory: Path) -> bool:
        """
        Check whether the memory exists in the specified directory.

        :param directory: The directory to check.
        :type directory: Path
        :return: True if the memory exists, False otherwise.
        :rtype: bool
        """
        pass

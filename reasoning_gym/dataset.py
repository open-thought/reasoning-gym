"""Base class for procedural dataset generators"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sized
from copy import deepcopy
from random import Random
from typing import Any, Dict, Iterator, Optional, Type, TypeVar


class ProceduralDataset(ABC, Sized, Iterable[Dict[str, Any]]):
    """Abstract base class for procedural dataset generators"""

    def __init__(self, config: Any, seed: Optional[int] = None, size: int = 500):
        """Initialize the dataset with config, optional seed and size"""
        if hasattr(config, "validate") and callable(config.validate):
            config.validate()

        self.config = config
        self.size = size
        self.seed = seed if seed is not None else Random().randint(0, 2**32)

    def __len__(self) -> int:
        """Return the virtual size of the dataset"""
        return self.size

    def __iter__(self):
        """Make the dataset iterable"""
        self._current_idx = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        """Get next item in iteration"""
        if self._current_idx >= self.size:
            raise StopIteration
        item = self[self._current_idx]
        self._current_idx += 1
        return item

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        """Generate a single dataset item

        Args:
            idx: Index of the item to generate

        Returns:
            dict containing at least:
                - question: str
                - answer: str
                - metadata: dict
        """
        raise NotImplementedError

    def score_answer(self, answer: Optional[str], entry: Dict[str, any]) -> float:
        """Overwrite this method in derived classes if a single oracle answer is not available."""
        oracle_answer = entry["answer"]
        reward = 0.0
        if answer is not None:
            if answer == oracle_answer:
                reward = 1.0
            elif oracle_answer in answer:
                reward = 0.5
            else:
                reward = 0.01

        return reward


T = TypeVar("T", bound="ProceduralDataset")


class ReseedingDataset(Iterable[Dict[str, Any]]):
    """Wrapper that makes any ProceduralDataset infinite by reseeding when reaching the end"""

    def __init__(self, dataset: T, chunk_size: int = 500):
        """Initialize with dataset instance and chunk size

        Args:
            dataset: The ProceduralDataset instance to wrap
            chunk_size: Size of each generated chunk before reseeding
        """
        self.dataset = dataset
        self.dataset_cls: Type[T] = type(dataset)
        self.chunk_size = chunk_size

        # Start with chunk 0
        self._current_chunk = 0
        self._current_dataset = self._create_chunk(0)
        self._current_idx = 0

    def _create_chunk(self, chunk_num: int) -> T:
        """Create a new dataset chunk with unique seed"""
        # Create new config with modified seed
        new_config = deepcopy(self.dataset.config)
        if hasattr(new_config, "seed"):
            # Derive new seed from chunk number using dataset's seed, wrapping around at 2^32
            new_config.seed = (self.dataset.seed + chunk_num) % (2**32)

        # Create new dataset instance with chunk config
        return self.dataset_cls(new_config)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Make the dataset iterable"""
        self._current_chunk = 0
        self._current_dataset = self._create_chunk(0)
        self._current_idx = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        """Get next item, creating new chunk if needed"""
        if self._current_idx >= self.chunk_size:
            # Move to next chunk
            self._current_chunk += 1
            self._current_dataset = self._create_chunk(self._current_chunk)
            self._current_idx = 0

        item = self._current_dataset[self._current_idx]
        self._current_idx += 1
        return item

    def score_answer(self, answer: Optional[str], entry: Dict[str, any]) -> float:
        """Forward scoring to the wrapped dataset's implementation"""
        return self.dataset.score_answer(answer, entry)

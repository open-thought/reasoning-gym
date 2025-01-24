"""Base class for procedural dataset generators"""
from abc import ABC, abstractmethod
from random import Random
from typing import Optional


class ProceduralDataset(ABC):
    """Abstract base class for procedural dataset generators"""
    
    def __init__(self, seed: Optional[int] = None, size: int = 500):
        """Initialize the dataset with optional seed and size"""
        self.size = size
        self.seed = seed if seed is not None else Random().randint(0, 2**32)
        
    def __len__(self) -> int:
        """Return the virtual size of the dataset"""
        return self.size
    
    def __iter__(self):
        """Make the dataset iterable"""
        self._current_idx = 0
        return self
    
    def __next__(self):
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

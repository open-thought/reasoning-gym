"""GSM Symblic dataset generator"""

from . import generators
from dataclasses import dataclass
from random import Random
from typing import List, Optional

from reasoning_gym.factory import ProceduralDataset, register_dataset

@dataclass
class GSMSymbolicDatasetConfig:
    """Configuration for GSM symbolic task generation"""
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        pass

class GSMSymbolicDataset(ProceduralDataset):

    def __init__(self, config, seed = None, size = 500):
        super().__init__(config, seed, size)
        # Initialize as None
        self._generators = None

    @property
    def generators(self):
        """Lazy load generators only when first accessed"""
        if self._generators is None:
            self._generators = self.get_generators()
        return self._generators

    def get_generators(self):
        """
        Generates mapper from task identifiers (keys) to example generator functions
        """
        prefix = 'generate_'
        return {
            self.strip_prefix(n, prefix): getattr(generators, n) for n in dir(generators) if n.startswith(prefix)
        }

    def strip_prefix(self, s, prefix):
        return s[len(prefix):]

    def __getitem__(self, idx) -> dict:
        """Generate a single GSM symbolic dataset"""
        rng = Random(self.seed + idx)
        # Stringify the random integer generated from the random number generator
        generator_idx = str(rng.randint(0, len(self.generators) - 1))
        generator = self.generators[generator_idx]
        # Here the res is a dictionary of 
        res = generator(rng)
        return res
    
register_dataset("gsm_symbolic", GSMSymbolicDataset, GSMSymbolicDatasetConfig)
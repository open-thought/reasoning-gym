"""
Arithmetic tasks for training reasoning capabilities:
- Basic arithmetic
- Chain sums
- Word problems
- Leg counting
"""

from .basic_arithmetic import ArithmeticDataset, ArithmeticDatasetConfig, arithmetic_dataset
from .chain_sum import ChainSum, ChainSumConfig, chain_sum_dataset
from .leg_counting import LegCountingConfig, LegCountingDataset, leg_counting_dataset
from .prime_factorization import PrimeFactorizationConfig, PrimeFactorizationDataset, prime_factorization_dataset

__all__ = [
    "ArithmeticDataset",
    "ArithmeticDatasetConfig",
    "arithmetic_dataset",
    "ChainSum",
    "ChainSumConfig",
    "chain_sum_dataset",
    "LegCountingConfig",
    "LegCountingDataset",
    "leg_counting_dataset",
    "PrimeFactorizationConfig",
    "PrimeFactorizationDataset",
    "prime_factorization_dataset"
]

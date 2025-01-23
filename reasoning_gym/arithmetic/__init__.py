"""
Arithmetic tasks for training reasoning capabilities:
- Basic arithmetic
- Chain sums
- Word problems
- Leg counting
"""

from .basic_arithmetic import ArithmeticDataset, ArithmeticDatasetConfig, arithmetic_dataset
from .chain_sum import ChainSum, ChainSumConfig, chain_sum
from .leg_counting import LegCountingConfig, LegCountingDataset, leg_counting_dataset

__all__ = [
    "ArithmeticDataset",
    "ArithmeticDatasetConfig",
    "arithmetic_dataset",
    "ChainSum",
    "ChainSumConfig",
    "chain_sum",
    "LegCountingConfig",
    "LegCountingDataset",
    "leg_counting_dataset"
]

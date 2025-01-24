"""
Arithmetic tasks for training reasoning capabilities:
- Basic arithmetic
- Chain sums
- Word problems
- Leg counting
"""

from .basic_arithmetic import BasicArithmeticDataset, BasicArithmeticDatasetConfig, basic_arithmetic_dataset
from .chain_sum import ChainSum, ChainSumConfig, chain_sum_dataset
from .fraction_simplification import (
    FractionSimplificationConfig,
    FractionSimplificationDataset,
    fraction_simplification_dataset,
)
from .gcd import GCDConfig, GCDDataset, gcd_dataset
from .lcm import LCMConfig, LCMDataset, lcm_dataset
from .leg_counting import LegCountingConfig, LegCountingDataset, leg_counting_dataset
from .prime_factorization import PrimeFactorizationConfig, PrimeFactorizationDataset, prime_factorization_dataset

__all__ = [
    "BasicArithmeticDataset",
    "BasicArithmeticDatasetConfig",
    "basic_arithmetic_dataset",
    "ChainSum",
    "ChainSumConfig",
    "chain_sum_dataset",
    "FractionSimplificationConfig",
    "FractionSimplificationDataset",
    "fraction_simplification_dataset",
    "GCDConfig",
    "GCDDataset",
    "gcd_dataset",
    "LCMConfig",
    "LCMDataset",
    "lcm_dataset",
    "LegCountingConfig",
    "LegCountingDataset",
    "leg_counting_dataset",
    "PrimeFactorizationConfig",
    "PrimeFactorizationDataset",
    "prime_factorization_dataset",
]

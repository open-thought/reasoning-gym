"""
Arithmetic tasks for training reasoning capabilities:
- Basic arithmetic
- Chain sums
- Word problems
- Leg counting
"""

from .basic_arithmetic import BasicArithmeticDataset, BasicArithmeticDatasetConfig
from .chain_sum import ChainSum, ChainSumConfig
from .fraction_simplification import FractionSimplificationConfig, FractionSimplificationDataset
from .gcd import GCDConfig, GCDDataset
from .lcm import LCMConfig, LCMDataset
from .leg_counting import LegCountingConfig, LegCountingDataset
from .prime_factorization import PrimeFactorizationConfig, PrimeFactorizationDataset
from .gsm_symbolic.gsm_symbolic_datasets import GSMSymbolicDataset, GSMSymbolicDatasetConfig

__all__ = [
    "BasicArithmeticDataset",
    "BasicArithmeticDatasetConfig",
    "basic_arithmetic_dataset",
    "ChainSum",
    "ChainSumConfig",
    "FractionSimplificationConfig",
    "FractionSimplificationDataset",
    "GCDConfig",
    "GCDDataset",
    "LCMConfig",
    "LCMDataset",
    "LegCountingConfig",
    "LegCountingDataset",
    "PrimeFactorizationConfig",
    "PrimeFactorizationDataset",
    "GSMSymbolicDatasetConfig",
    "GSMSymbolicDataset",
]

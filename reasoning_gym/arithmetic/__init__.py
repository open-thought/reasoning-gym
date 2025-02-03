"""
Arithmetic tasks for training reasoning capabilities:
- Basic arithmetic
- Chain sums
- Word problems
- Leg counting
- Time intervals
"""

from .basic_arithmetic import BasicArithmeticDataset, BasicArithmeticDatasetConfig
from .calendar_arithmetic import CalendarArithmeticConfig, CalendarArithmeticDataset
from .chain_sum import ChainSum, ChainSumConfig
from .fraction_simplification import FractionSimplificationConfig, FractionSimplificationDataset
from .gcd import GCDConfig, GCDDataset
from .gsm_symbolic.gsm_symbolic_datasets import GSMSymbolicDataset, GSMSymbolicDatasetConfig
from .lcm import LCMConfig, LCMDataset
from .leg_counting import LegCountingConfig, LegCountingDataset
from .prime_factorization import PrimeFactorizationConfig, PrimeFactorizationDataset
from .time_intervals import TimeIntervalsConfig, TimeIntervalsDataset

__all__ = [
    "BasicArithmeticDataset",
    "BasicArithmeticDatasetConfig",
    "basic_arithmetic_dataset",
    "ChainSum",
    "ChainSumConfig",
    "CalendarArithmeticConfig",
    "CalendarArithmeticDataset",
    "Weekday",
    "CalendarTask",
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
    "TimeIntervalsConfig",
    "TimeIntervalsDataset",
]

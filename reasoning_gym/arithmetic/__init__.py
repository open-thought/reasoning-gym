"""
Arithmetic tasks for training reasoning capabilities:
- Basic arithmetic
- Chain sums
- Word problems
- Leg counting
- Time intervals
"""

from .basic_arithmetic import BasicArithmeticDataset
from .calendar_arithmetic import CalendarArithmeticDataset
from .chain_sum import ChainSumDataset
from .fraction_simplification import FractionSimplificationDataset
from .gcd import GcdDataset
from .lcm import LcmDataset
from .leg_counting import LegCountingDataset
from .prime_factorization import PrimeFactorizationDataset
from .time_intervals import TimeIntervalsDataset

__all__ = [
    "BasicArithmeticDataset",
    "CalendarArithmeticDataset",
    "ChainSumDataset",
    "FractionSimplificationDataset",
    "GcdDataset",
    "LcmDataset",
    "LegCountingDataset",
    "PrimeFactorizationDataset",
    "TimeIntervalsDataset",
]

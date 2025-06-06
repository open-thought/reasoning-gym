"""
Arithmetic tasks for training reasoning capabilities:
"""

from .basic_arithmetic import BasicArithmeticCurriculum, BasicArithmeticDataset, BasicArithmeticDatasetConfig
from .bitwise_arithmetic import BitwiseArithmeticConfig, BitwiseArithmeticCurriculum, BitwiseArithmeticDataset
from .calendar_arithmetic import CalendarArithmeticConfig, CalendarArithmeticCurriculum, CalendarArithmeticDataset
from .chain_sum import ChainSumConfig, ChainSumDataset
from .count_bits import CountBitsConfig, CountBitsCurriculum, CountBitsDataset
from .decimal_arithmetic import DecimalArithmeticConfig, DecimalArithmeticCurriculum, DecimalArithmeticDataset
from .decimal_chain_sum import DecimalChainSumConfig, DecimalChainSumCurriculum, DecimalChainSumDataset
from .dice import DiceConfig, DiceCurriculum, DiceDataset
from .fraction_simplification import (
    FractionSimplificationConfig,
    FractionSimplificationCurriculum,
    FractionSimplificationDataset,
)
from .gcd import GCDConfig, GCDCurriculum, GCDDataset
from .gsm_symbolic.gsm_symbolic import GSMSymbolicDataset, GSMSymbolicDatasetConfig
from .lcm import LCMConfig, LCMCurriculum, LCMDataset
from .leg_counting import LegCountingConfig, LegCountingCurriculum, LegCountingDataset
from .number_format import NumberFormatConfig, NumberFormatCurriculum, NumberFormatDataset
from .power_function import PowerFunctionConfig, PowerFunctionCurriculum, PowerFunctionDataset
from .prime_factorization import PrimeFactorizationConfig, PrimeFactorizationCurriculum, PrimeFactorizationDataset
from .products import ProductsConfig, ProductsDataset
from .time_intervals import TimeIntervalsConfig, TimeIntervalsCurriculum, TimeIntervalsDataset

__all__ = [
    "BasicArithmeticDataset",
    "BasicArithmeticDatasetConfig",
    "BasicArithmeticCurriculum",
    "ChainSumDataset",
    "ChainSumConfig",
    "CalendarArithmeticConfig",
    "CalendarArithmeticDataset",
    "CalendarArithmeticCurriculum",
    "FractionSimplificationConfig",
    "FractionSimplificationDataset",
    "FractionSimplificationCurriculum",
    "GCDConfig",
    "GCDDataset",
    "GCDCurriculum",
    "LCMConfig",
    "LCMDataset",
    "LCMCurriculum",
    "LegCountingConfig",
    "LegCountingDataset",
    "LegCountingCurriculum",
    "PowerFunctionConfig",
    "PowerFunctionDataset",
    "PowerFunctionCurriculum",
    "PrimeFactorizationConfig",
    "PrimeFactorizationDataset",
    "PrimeFactorizationCurriculum",
    "ProductsDataset",
    "ProductsConfig",
    "GSMSymbolicDatasetConfig",
    "GSMSymbolicDataset",
    "TimeIntervalsConfig",
    "TimeIntervalsDataset",
    "TimeIntervalsCurriculum",
    "CountBitsConfig",
    "CountBitsDataset",
    "CountBitsCurriculum",
    "DiceConfig",
    "DiceDataset",
    "DiceCurriculum",
    "NumberFormatConfig",
    "NumberFormatDataset",
    "NumberFormatCurriculum",
    "DecimalArithmeticConfig",
    "DecimalArithmeticDataset",
    "DecimalArithmeticCurriculum",
    "DecimalChainSumCurriculum",
    "DecimalChainSumConfig",
    "DecimalChainSumDataset",
    "BitwiseArithmeticConfig",
    "BitwiseArithmeticDataset",
    "BitwiseArithmeticCurriculum",
]

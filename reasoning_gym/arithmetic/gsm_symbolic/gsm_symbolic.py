"""GSM Symblic dataset generator"""

from dataclasses import dataclass
from random import Random
from typing import Any, Callable, Optional

from reasoning_gym.factory import ProceduralDataset, register_dataset

tasks_ok = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    33,
    34,
    36,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    62,
    64,
    66,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    75,
    78,
    80,
    81,
    82,
    83,
    84,
    85,
    88,
    89,
    91,
    92,
    93,
    94,
    95,
    96,
    99,
]
tasks_need_fix = [32, 35, 37, 61, 63, 65, 74, 76, 77, 79, 86, 87, 90, 97, 98]


@dataclass
class GSMSymbolicDatasetConfig:
    """Configuration for GSM symbolic task generation"""

    difficulty: float = 1.0
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.size > 0, "size must be positive"
        assert 1.0 <= self.difficulty <= 1.0  # currently only difficulty 1.0 is supported


class GSMSymbolicDataset(ProceduralDataset):

    def __init__(self, config: GSMSymbolicDatasetConfig):
        super().__init__(config, config.seed, config.size)
        self._generators: dict[int, Callable[[Random, float], dict[str, Any]]] = None  # initially None, lazy loading
        self.task_indices = Random(self.seed).choices(tasks_ok, k=self.size)

    @property
    def generators(self) -> dict[int, Callable[[Random, float], dict[str, Any]]]:
        """Lazy load generators only when first accessed"""
        if self._generators is None:
            self._generators = self._load_generators()
        return self._generators

    def _load_generators(self):
        """
        Generates mapper from task identifiers (keys) to example generator functions
        """
        from . import generators_00_49, generators_50_99

        def strip_prefix(s: str, prefix: str) -> str:
            return s[len(prefix) :]

        prefix = "generate_"
        gs = {}
        for n in dir(generators_00_49):
            if n.startswith(prefix):
                gs[int(strip_prefix(n, prefix))] = getattr(generators_00_49, n)
        for n in dir(generators_50_99):
            if n.startswith(prefix):
                gs[int(strip_prefix(n, prefix))] = getattr(generators_50_99, n)
        return gs

    def __getitem__(self, idx: int) -> dict:
        """Generate a single GSM symbolic dataset"""
        rng = Random(self.seed + idx)
        generator_idx = self.task_indices[idx]
        generator = self.generators[generator_idx]
        example = generator(rng, self.config.difficulty)
        example["question"] += " Give only the result as your final answer."
        return example


register_dataset("gsm_symbolic", GSMSymbolicDataset, GSMSymbolicDatasetConfig)

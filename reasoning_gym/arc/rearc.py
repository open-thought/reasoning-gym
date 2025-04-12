from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable, Optional

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset
from .board_format import ARC_PROMPT_TEMPLATE, BoardFormattingOptions, format_board, format_board_pair, parse_board

RNG_DIFFICULTY_LEVELS = [0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]
RNG_DIFFICULTY_RANGES = [
    (RNG_DIFFICULTY_LEVELS[i], RNG_DIFFICULTY_LEVELS[i + 1]) for i in range(len(RNG_DIFFICULTY_LEVELS) - 1)
]

PSO_DIFFICULTY_LEVELS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 1]
PSO_DIFFICULTY_RANGES = [
    (PSO_DIFFICULTY_LEVELS[i], PSO_DIFFICULTY_LEVELS[i + 1]) for i in range(len(PSO_DIFFICULTY_LEVELS) - 1)
]

DATASET_NAME = "rearc"


@dataclass
class ReArcConfig:
    min_examples: int = 3  # minimum number of board pairs shown
    max_examples: int = 5  # maximum number of board pairs shown
    diff_lb: int = 0
    diff_ub: int = 0.2
    board_format_opts: BoardFormattingOptions = field(default_factory=lambda: BoardFormattingOptions())
    seed: Optional[int] = None
    size: int = 500
    rng_difficulty_ranges: list[tuple[float, float]] = field(default_factory=lambda: RNG_DIFFICULTY_RANGES)
    rng_difficulty_weights: list[float] = field(
        default_factory=lambda: [1 / len(RNG_DIFFICULTY_RANGES)] * len(RNG_DIFFICULTY_RANGES)
    )
    pso_difficulty_ranges: list[tuple[float, float]] = field(default_factory=lambda: PSO_DIFFICULTY_RANGES)
    pso_difficulty_weights: list[float] = field(
        default_factory=lambda: [1 / len(PSO_DIFFICULTY_RANGES)] * len(PSO_DIFFICULTY_RANGES)
    )

    def validate(self):
        assert self.min_examples > 0, "min_examples must be positive"
        assert self.min_examples <= self.max_examples, "min_examples must be <= max_examples"
        assert self.diff_lb <= self.diff_ub, "diff_lb must be <= diff_ub."
        assert self.size > 0, "Size of dataset must be positive."
        assert len(self.rng_difficulty_ranges) == len(
            self.rng_difficulty_weights
        ), "rng_difficulty_ranges and rng_difficulty_weights must have the same length."
        assert len(self.pso_difficulty_ranges) == len(
            self.pso_difficulty_weights
        ), "pso_difficulty_ranges and pso_difficulty_weights must have the same length."


class ReArcDataset(ProceduralDataset):
    def __init__(self, config: ReArcConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.board_format_opts = config.board_format_opts
        self._prompt_templates = ARC_PROMPT_TEMPLATE
        self.diff_lb = config.diff_lb
        self.diff_ub = config.diff_ub

        # lazy import of re-arc dsl & generators
        from .rearc_utils import generators
        from .rearc_utils.utils import get_generators, get_pso_difficulty

        self._generators = get_generators(generators)
        self.get_pso_difficulty = get_pso_difficulty

    @staticmethod
    def get_rng_difficulty(rng: Random) -> float:
        if not hasattr(rng, "difficulty_samples"):
            return 0.0
        samples = rng.difficulty_samples
        avg = sum(samples) / len(samples) if samples else 0.0
        rng.difficulty_samples = []
        return avg

    def __len__(self) -> int:
        return self.size

    def format_rearc_input(self, rng: Random, task: dict, generator: Callable) -> str:
        """
        Format a ReArc task input with multiple examples and test input.
        """

        num_examples = rng.randint(self.config.min_examples, self.config.max_examples)
        examples = [
            format_board_pair(
                i + 1, generator(rng, self.diff_lb, self.diff_ub), formatting_options=self.config.board_format_opts
            )
            for i in range(num_examples)
        ]
        examples = "".join(examples)
        input_grid = format_board(task["input"], self.board_format_opts)

        return self._prompt_templates.format(examples=examples, input_grid=input_grid)

    def __getitem__(self, idx: int) -> dict:
        """
        Generate a single ReArc task
        """
        rng = Random(self.seed + idx)

        pso_difficulty_range = rng.choices(
            self.config.pso_difficulty_ranges, weights=self.config.pso_difficulty_weights, k=1
        )[0]

        while True:
            task_id = rng.choice(list(self._generators.keys()))
            generator = self._generators[task_id]
            difficulty_range = rng.choices(
                self.config.rng_difficulty_ranges, weights=self.config.rng_difficulty_weights, k=1
            )[0]
            task = generator(rng, difficulty_range[0], difficulty_range[1])
            pso_difficulty = self.get_pso_difficulty(task)
            if (pso_difficulty_range[0] <= pso_difficulty) and (pso_difficulty <= pso_difficulty_range[1]):
                break

        rng_difficulty = self.get_rng_difficulty(rng)
        input_prompt = self.format_rearc_input(rng, task, generator)
        answer = format_board(task["output"], self.board_format_opts)

        return {
            "question": input_prompt,
            "answer": answer,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "input": task["input"],
                "output": task["output"],
                "task_id": task_id,
                "rng": rng_difficulty,
                "pso": pso_difficulty,
                "difficulty": {
                    "rng_difficulty_weights": self.config.rng_difficulty_weights,
                    "pso_difficulty_weights": self.config.pso_difficulty_weights,
                },
            },
        }

    def score_answer(self, answer: str, entry: dict[str, Any]) -> float:
        reward = 0.0
        metadata = entry["metadata"]
        if answer is not None:
            try:
                answer_board = parse_board(answer, self.board_format_opts)
                if answer_board == metadata["output"]:
                    reward = 1.0
                else:
                    reward = 0.05
            except:
                reward = 0.0
        return reward


class ReArcCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(ReArcCurriculum.__name__, ReArcConfig)
        self._define_attributes(
            ScalarAttributeDefinition(
                name="pso_difficulty_weights",
                field_name="pso_difficulty_weights",
                description="The range of PSO difficulty for the Arc problem",
                levels=[
                    [1, 0, 0, 0, 0, 0, 0],  # only sample/generate the easiest tasks wrs PSO difficulty
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                ],  # only sample/generate the hardest tasks PSO difficulty
            ),
            ScalarAttributeDefinition(
                name="rng_difficulty_weights",
                field_name="rng_difficulty_weights",
                description="The range of RNG difficulty for the Arc problem",
                levels=[
                    [1, 0, 0, 0, 0, 0, 0],  # only sample/generate the easiest tasks wrs RNG difficulty
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                ],  # only sample/generate the hardest tasks wrs RNG difficulty
            ),
        )


register_dataset(DATASET_NAME, ReArcDataset, ReArcConfig, ReArcCurriculum)

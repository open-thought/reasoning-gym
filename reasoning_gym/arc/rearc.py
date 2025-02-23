from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable, Optional

from ..factory import ProceduralDataset, register_dataset
from .board_format import ARC_PROMPT_TEMPLATE, BoardFormattingOptions, format_board, format_board_pair, parse_board


@dataclass
class ReArcConfig:
    min_examples: int = 3  # minimum number of board pairs shown
    max_examples: int = 5  # maximum number of board pairs shown
    diff_lb: int = 0
    diff_ub: int = 0.2
    board_format_opts: BoardFormattingOptions = field(default_factory=lambda: BoardFormattingOptions())
    seed: Optional[int] = None
    size: int = 500

    def validate(self):
        assert self.min_examples > 0, "min_examples must be positive"
        assert self.min_examples <= self.max_examples, "min_examples must be <= max_examples"
        assert self.diff_lb <= self.diff_ub, "diff_lb must be <= diff_ub."
        assert self.size > 0, "Size of dataset must be positive."


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
        task_id = rng.choice(list(self._generators.keys()))
        generator = self._generators[task_id]
        task = generator(rng, self.diff_lb, self.diff_ub)

        rng_difficulty = self.get_rng_difficulty(rng)
        pso_difficulty = self.get_pso_difficulty(task)
        input_prompt = self.format_rearc_input(rng, task, generator)
        answer = format_board(task["output"], self.board_format_opts)

        return {
            "question": input_prompt,
            "answer": answer,
            "metadata": {
                "input": task["input"],
                "output": task["output"],
                "task_id": task_id,
                "difficulty": {
                    "rng": rng_difficulty,
                    "pso": pso_difficulty,
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
                reward = 0.01
        return reward


register_dataset("rearc", ReArcDataset, ReArcConfig)

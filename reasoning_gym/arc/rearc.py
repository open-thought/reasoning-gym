from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable, Dict, Optional

from ..factory import ProceduralDataset, register_dataset
from .rearc_board_format import BoardFormattingOptions, default_board_format_opts, format_board, format_board_pair
from .rearc_utils import generators, verifiers
from .rearc_utils.dsl import *
from .rearc_utils.utils import *

_REARC_PROMPT_TEMPLATES = """Find the common rule that maps an input grid to an output grid, given the examples below

Examples:
{examples}

Below is a test input grid. Predict the corresponding output grid by applying the rule you found.
Your final answer should just be the text output grid itself.


Input Grid:
{input_grid}

Output Grid:"""

_COLOUR_MAP = ListedColormap(
    ["#000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25"]
)


def strip_prefix(string: str, prefix: str) -> str:
    """
    removes prefix
    """
    return string[len(prefix) :]


def get_generators() -> dict:
    """
    returns mapper from task identifiers (keys) to example generator functions
    """
    prefix = "generate_"
    return {strip_prefix(n, prefix): getattr(generators, n) for n in dir(generators) if n.startswith(prefix)}


def get_verifiers() -> dict:
    """
    returns mapper from task identifiers (keys) to example verifier functions
    """
    prefix = "verify_"
    return {strip_prefix(n, prefix): getattr(verifiers, n) for n in dir(verifiers) if n.startswith(prefix)}


@dataclass
class ReArcConfig:
    diff_lb: int = 0
    diff_ub: int = 1
    board_format_opts: BoardFormattingOptions = field(default_factory=default_board_format_opts)
    seed: Optional[int] = None
    size: int = 500

    def validate(self):
        assert self.diff_lb < self.diff_ub, "diff_lb must be < diff_ub."
        assert self.size > 0, "Size of dataset must be positive."


class ReArcDataset(ProceduralDataset):
    def __init__(self, config: ReArcConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.board_format_opts = config.board_format_opts
        self._prompt_templates = _REARC_PROMPT_TEMPLATES
        self.diff_lb = config.diff_lb
        self.diff_ub = config.diff_ub
        self._generators = get_generators()
        self._verifiers = get_verifiers()

    @staticmethod
    def get_rng_difficulty(rng: Random) -> float:
        if not hasattr(rng, "difficulty_samples"):
            return 0.0
        samples = rng.difficulty_samples
        avg = sum(samples) / len(samples) if samples else 0.0
        rng.difficulty_samples = []
        return avg

    @staticmethod
    def get_pso_difficulty(example: dict) -> float:
        """
        PSO-Difficulty: proxy measure for example difficulty, defined as weighted sum of #Pixels, #Symbols, #Objects
        """
        i, o = example["input"], example["output"]
        hwi = height(i) * width(i)
        hwo = height(o) * width(o)
        pix_pct = (hwi + hwo) / 1800
        col_pct = len(palette(i) | palette(o)) / 10
        obj_dens = (len(objects(i, T, F, F)) / hwi + len(objects(o, T, F, F)) / hwo) / 2
        return (pix_pct + col_pct + obj_dens) / 3

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def visualise_pair(example: Dict[str, Any]) -> None:
        """
        Visualise a ReArc task pair
        """
        norm = Normalize(vmin=0, vmax=9)
        args = {"cmap": _COLOUR_MAP, "norm": norm}

        # Change to 1 row, 2 columns
        height = 1
        width = 2
        figure_size = (3 * width * 3, height * 3)
        figure, axes = plt.subplots(height, width, figsize=figure_size)

        # Plot input and output side by side
        axes[0].imshow(example["metadata"]["input"], **args)
        axes[1].imshow(example["metadata"]["output"], **args)

        # Add titles to distinguish the plots
        axes[0].set_title("Input")
        axes[1].set_title("Output")

    def format_rearc_input(self, idx: int, task: dict, generator: Callable) -> str:
        """
        Format a ReArc task input with multiple examples and test input.
        """
        example_1 = generator(Random((self.seed + idx) * 1 * self.size), self.diff_lb, self.diff_ub)
        example_2 = generator(Random((self.seed + idx) * 2 * self.size), self.diff_lb, self.diff_ub)
        example_3 = generator(Random((self.seed + idx) * 3 * self.size), self.diff_lb, self.diff_ub)

        examples = (
            format_board_pair(1, example_1, self.board_format_opts)
            + format_board_pair(2, example_2, self.board_format_opts)
            + format_board_pair(3, example_3, self.board_format_opts)
        )
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
        input_prompt = self.format_rearc_input(idx, task, generator)

        return {
            "question": input_prompt,
            "answer": task["output"],
            "metadata": {
                "input": task["input"],
                "output": task["output"],
                "task_id": task_id,
                "rng": rng_difficulty,
                "pso": pso_difficulty,
            },
        }

    def score_answer(self, answer: str, metadata: Dict[str, Any]) -> float:
        reward = 0.0
        if answer is not None:
            try:
                task_id = metadata["task_id"]
                verifier = self._verifiers[task_id]
                if verifier(metadata["input"]) == metadata["output"]:
                    reward = 1.0
                else:
                    reward = 0.05
            except:
                reward = 0.01
        return reward


register_dataset("rearc", ReArcDataset, ReArcConfig)

from dataclasses import dataclass
from random import Random

from ..factory import ProceduralDataset, register_dataset
from .rearc_board_format import BoardFormattingOptions, format_arc_task, format_board, format_board_pair
from .rearc_utils import generators, verifiers


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
    seed: Optional[int] = None
    size: int = 500
    board_format_opts: BoardFormattingOptions


REARC_PROMPT_TEMPLATES = """Find the common rule that maps an input grid to an output grid, given the examples below

Examples:
{examples}

Below is a test input grid. Predict the corresponding output grid by applying the rule you found.
Your final answer should just be the text output grid itself.


Input Grid:
{input_grid_test}

Output Grid:"""


class ReArcDataset(ProceduralDataset):
    def __init__(self, config: ReArcConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        board_format_opts = config.board_format_opts
        self._prompt_templates = REARC_PROMPT_TEMPLATES
        self._generators = get_generators()
        self._verifiers = get_verifiers()

    @staticmethod
    def get_rng_difficulty(example: dict) -> float:
        if not hasattr(rng, "difficulty_samples"):
            return 0.0
        samples = rng.difficulty_samples
        avg = sum(samples) / len(samples) if samples else 0.0
        rng.difficulty_samples = []  # Reset for next generation
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

    def format_rearc_input(self, idx: int, task: dict, generator: Callable) -> str:
        """
        Format a ReArc task input with multiple examples and test input.
        """
        example_1 = generator(Random((self.seed + idx) * 1 * self.size))
        example_2 = generator(Random((self.seed + idx) * 2 * self.size))
        example_3 = generator(Random((self.seed + idx) * 3 * self.size))

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
        task = generator(rng)

        input_grid = format_board(task["input"], self.board_format_opts)
        output_grid = format_board(task["output"], self.board_format_opts)

        rng_difficulty = self.get_rng_difficulty(rng)
        pso_difficulty = self.get_pso_difficulty(task)
        input_prompt = self.format_rearc_input(idx, task, generator)

        return {
            "question": input_prompt,
            "answer": task["output"],
            "metadata": {"difficulty": {"output_grid": output_grid, "rng": rng_difficulty, "pso": pso_difficulty}},
        }

    def score_answer(self, answer: str, metadata: Dict[str, Any]) -> float:
        """Todo"""


dataset = register_dataset("rearc", ReArcDataset)

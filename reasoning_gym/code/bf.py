from dataclasses import dataclass
from random import Random
from typing import Dict, Optional

import bfi

from ..data.wordle_words import wordle_words
from ..factory import ProceduralDataset, register_dataset
from .contrib.bfit.Compiler import Compiler, Minify


@dataclass
class BFConfig:
    """Configuration for BF task generation"""

    seed: Optional[int] = None
    size: int = 500
    difficulty: int = 1

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.difficulty > 0, "difficulty must be greater than 0"
        assert self.difficulty < 4, "difficulty must be less than 4"


class BFDataset(ProceduralDataset):
    """Generates BF tasks"""

    def __init__(self, config: BFConfig):
        self._prompt_templates = [
            "This is a BF (Brainf*ck) computer program. What is the output? \n\n{bf_program}",
        ]
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single BF task

        Returns:
            dict with keys:
                - question: str, the task description with BF program
                - answer: str, the result of this BF program BFI execution
                - metadata: dict with generation parameters
        """
        rng = Random(self.seed + idx)

        bfit_code = self.generate_bfit_code(self.config.difficulty, rng)
        bf_program = self.compile_bfit_code_to_bf(bfit_code)
        result = bfi.interpret(bf_program, buffer_output=True)

        return {
            "question": rng.choice(self._prompt_templates).format(bf_program=bf_program),
            "answer": result,
            "metadata": {"bfit_code": bfit_code, "bf_program": bf_program},
        }

    def generate_bfit_code(self, difficulty, rng: Random) -> str:

        if difficulty == 1:
            word = rng.choice(wordle_words)
            bfit_template = f"""
int main() {{
    print("{word}");
}}
"""
        elif difficulty == 2:
            x = rng.randint(1, 4)
            y = rng.randint(1, 5)
            target = x * y * rng.randint(1, 9) + rng.randint(1, 9)
            bfit_template = f"""
int main() {{
    int acc = 0;
    int target = {target};
    int x = {x};
    int y = {y};
    while (acc < target) {{
        acc = acc + x;
        acc = acc + y;
    }}
    printint(acc);
}}
"""
        elif difficulty == 3:
            x = rng.randint(1, 7)
            y = rng.randint(1, 9)
            target = x * y * rng.randint(1, 9) + rng.randint(1, 9) + 50
            conditional = target - rng.randint(1, 40)
            bfit_template = f"""
int main() {{
    int acc = 0;
    int target = {target};
    int x = {x};
    int y = {y};
    while (acc < target) {{
        acc = acc + x;
        if (acc > {conditional}) {{
            acc = acc + y;
        }}
    }}
    printint(acc);
}}
"""
        rendered_bfit = bfit_template
        return rendered_bfit

    def compile_bfit_code_to_bf(self, bfit: str) -> str:
        bf = Compiler.compile(bfit, optimize_code=True)
        # bf = Minify.minify(bf) # Is this necessary?
        return bf

    def score_answer(self, answer: Optional[str], entry: Dict[str, any]) -> float:
        """Determine if the solution provided solves the BF task.

        The function awards 1.0 for a correct answer.

        Args:
            answer (Optional[str]): The user's answer.
            entry (Dict[str, any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        if answer == None:
            return 0.0
        if answer != entry["answer"]:
            return 0.01
        else:
            return 1.0  # Yay


# Register the dataset
register_dataset("bf", BFDataset, BFConfig)

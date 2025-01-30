from dataclasses import dataclass
from random import Random
from typing import Dict, Optional

import bfi
from .contrib.bfit.Compiler import Compiler, Minify

from ..data.wordle_words import wordle_words
from ..factory import ProceduralDataset, register_dataset


@dataclass
class BFConfig:
    """Configuration for BF task generation"""

    seed: Optional[int] = None
    size: int = 500


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
                - question: str, the task description with figlet string
                - answer: str, the figlet encoded word
                - metadata: dict with generation parameters
        """
        rng = Random(self.seed + idx)

        bfit_code = self.generate_bfit_code(rng)
        bf_program = self.compile_bfit_code_to_bf(bfit_code)

        result = bfi.interpret(bf_program, buffer_output=True)

        return {
            "question": rng.choice(self._prompt_templates).format(bf_program=bf_program),
            "answer": result,
            "metadata": {"bfit_code": bfit_code, "bf_program": bf_program},
        }

    def generate_bfit_code(self, rng: Random) -> str:

        bfit_template = """
int main() {
    int acc = 0;
    int target = 15;
    int x = 2;
    int y = 3;
    while (acc < target) {
        acc = acc + x;
        acc = acc + y;
    }
    printint(acc);
}
"""
        rendered_bfit = bfit_template
        return rendered_bfit

    def compile_bfit_code_to_bf(self, bfit: str) -> str:
        bf = Compiler.compile(bfit, optimize_code=True)
        # bf = Minify.minify(bf) # Is this necessary?
        return bf

    def score_answer(self, answer: Optional[str], entry: Dict[str, any]) -> float:
        """Determine if the solution provided solves the figlet task.

        The function awards 1.0 for a correct answer and 0.1 points for each correct letter in the correct position,
        with a maximum possible score of 1.0.

        Args:
            answer (Optional[str]): The user's answer.
            entry (Dict[str, any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        if answer == None:
            return 0.0
        if answer != entry['answer']:
            return 0.01
        else:
            return 1.0 # Yay

# Register the dataset
register_dataset("figlet_font", BFDataset, BFConfig)

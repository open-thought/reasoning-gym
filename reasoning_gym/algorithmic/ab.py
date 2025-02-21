from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset


def generate_program(length, rng):
    """Generates a random initial program of a given length."""
    elements = ["A#", "B#", "#A", "#B"]
    return [rng.choice(elements) for _ in range(length)]


def compute_steps(program, max_steps=100):
    """Computes the transformation steps and detects if the program does not halt."""
    steps = [program.copy()]
    seen_states = {tuple(program)}

    for step in range(max_steps):
        current = steps[-1]
        new_program = None

        for i in range(len(current) - 1):
            a, b = current[i], current[i + 1]
            if a == "A#" and b == "#A":
                new_program = current[:i] + current[i + 2 :]
            elif a == "A#" and b == "#B":
                new_program = current[:i] + ["#B", "A#"] + current[i + 2 :]
            elif a == "B#" and b == "#A":
                new_program = current[:i] + ["#A", "B#"] + current[i + 2 :]
            elif a == "B#" and b == "#B":
                new_program = current[:i] + current[i + 2 :]

            if new_program is not None:
                break

        if new_program is None:
            # No more transformations possible
            return steps, False

        if tuple(new_program) in seen_states:
            # Detected a loop, meaning non-halting behavior
            return steps, True

        steps.append(new_program)
        seen_states.add(tuple(new_program))

    return steps, True  # Reached max steps, assume non-halting


@dataclass
class ABConfig:
    """Configuration for A::B task generation"""

    seed: Optional[int] = None
    size: int = 500
    length: int = 10

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.length > 0, "length must be greater than 0"
        assert self.size > 0, "size must be greater than 0"


class ABDataset(ProceduralDataset):
    """Generates A::B tasks, as described by @VictorTaelin [here](https://x.com/VictorTaelin/status/1776096481704804789)"""

    def __init__(self, config: ABConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single AB task

        Returns:
            dict with keys:
                - question: str, the task description with AB program
                - answer: str, the result of this AB program ABI execution
                - metadata: dict with generation parameters
        """
        rng = Random(self.seed + idx)

        while True:
            initial_program = generate_program(self.config.length, rng)
            steps, non_halting = compute_steps(initial_program)
            if not non_halting:
                break

        # Via:
        #   https://x.com/VictorTaelin/status/1776248021858111542
        #   https://gist.github.com/VictorTaelin/e514844f4df9e5f182b28e5a07e44b17
        prompt = f"""A::B is a system with 4 tokens: `A#`, `#A`, `B#` and `#B`.

An A::B program is a sequence of tokens. Example:

    B# A# #B #A B#

To *compute* a program, we must rewrite neighbor tokens, using the rules:

    A# #A ... becomes ... nothing
    A# #B ... becomes ... #B A#
    B# #A ... becomes ... #A B#
    B# #B ... becomes ... nothing

In other words, whenever two neighbor tokens have their '#' facing each-other,
they must be rewritten according to the corresponding rule. For example, the
first example shown here is computed as:

    B# A# #B #A B# =
    B# #B A# #A B# =
    A# #A B# =
    B#

The steps were:
1. We replaced `A# #B` by `#B A#`.
2. We replaced `B# #B` by nothing.
3. We replaced `A# #A` by nothing.
The final result was just `B#`.

Now, consider the following program:

{' '.join(initial_program)}

Return the final state of the program.
"""

        return {
            "question": prompt,
            "answer": " ".join(steps[-1]),
            "metadata": {},
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the AB task.

        The function awards 1.0 for a correct answer.

        Args:
            answer (Optional[str]): The user's answer.
            entry (dict[str, Any]): The original dataset entry containing the correct answer.

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
register_dataset("ab", ABDataset, ABConfig)

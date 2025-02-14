"""Iteratively synthesize new machines and parts from existing ones using a set of rules.

https://github.com/yongchao98/CodeSteer-v1.0/blob/main/create_dataset/create_dataset_string_splitting.py
"""

from dataclasses import dataclass
from random import Random
from typing import Optional

from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """There is a dismantling engineer who has old machines A, B, and C.
He discovered that he can obtain a batch of new parts X, Y, Z through the following rules:
1. One unit of machine A can be dismanteled into two units of part X and one unit of part Y.
2. Two units of machine B can be dismanteled into one unit of part X.
3. Two units of machine C can be dismanteled into one unit of part Y.
4. One unit of machine B and one unit of machine C can be combined into one unit of machine A.
5. One unit of part X and one unit of part Y can be combined into one unit of part Z.

Given a certain number of initial machines, your job is to continuously cycle through the rules 1-5 above, exausting one rule at a time, until no more rules can be applied, or until a state (counts of each machine and part type) is repeated.
After you make use of a rule, you should update the counts of each machine and part type accordingly, and then restart the process from rule 1.

The output should be the count of each machine and part type after the rules have been exhaustively applied in the following order: A B C X Y Z.
For example 1 0 1 5 4 3 means that you have 1 machine A, 0 machine B, 1 machine C, 5 part X, 4 part Y, and 3 part Z.

Example:
- Input: You have 2 machines A, 0 machines B, and 1 machine C.
- Output: 0 0 1 2 0 2
- Explanation
    0. Initial state: 2 0 1 0 0 0
    1. We can apply rule 1 and trade 1 machine A for 2 part X and 1 part Y: 1 0 1 2 1 0
    2. Starting over, we can apply rule 1 again: 0 0 1 4 2 0
    3. In the next iteration, we can apply rule 5 and trade 1 part X and 1 part Y for 1 part Z: 0 0 1 3 1 1
    4. In the next iteration, we can apply rule 5 again: 0 0 1 2 0 2
    5. We can't apply any more rules, so the final answer is 0 0 1 2 0 2

Now, you have {A_machine} machine A, {B_machine} machine B, and {C_machine} machine C. Provide the count of each machine and part type after applying the above rules.
"""


@dataclass
class StringSplittingConfig:
    """Configuration for String Splitting dataset generation"""

    min_initial_machines: int = 0  # Minimum number of initial machines
    max_initial_machines: int = 5  # Maximum number of initial machines
    max_iterations: int = 1_000  # Maximum number of iterations to apply the rules (Safety check for infinite loops)

    size: int = 500  # Virtual dataset size
    seed: Optional[int] = None

    def validate(self):
        """Validate configuration parameters"""
        assert 0 <= self.min_initial_machines, "min_initial_machines must be non-negative"
        assert (
            self.min_initial_machines <= self.max_initial_machines
        ), "min_initial_machines must be less than or equal to max_initial_machines"
        assert 0 < self.max_iterations, "max_iterations must be positive"


class StringSplittingDataset(ProceduralDataset):
    """Generates String Splitting exercises with configurable difficulty"""

    def __init__(self, config: StringSplittingConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _apply_rule(self, counts: list[int]) -> list[int]:
        """
        Apply the first applicable rule to the given counts.
        In case no rule is applicable, the counts are returned unchanged.
        """
        # label the indices for the counts
        A, B, C, X, Y, Z = range(6)

        # Rule 1: A -> 2X + Y
        if counts[A] >= 1:
            counts[A] -= 1
            counts[X] += 2
            counts[Y] += 1
        # Rule 2: 2B -> X
        elif counts[B] >= 2:
            counts[B] -= 2
            counts[X] += 1
        # Rule 3: 2C -> Y
        elif counts[C] >= 2:
            counts[C] -= 2
            counts[Y] += 1
        # Rule 4: B + C -> A
        elif counts[B] >= 1 and counts[C] >= 1:
            counts[B] -= 1
            counts[C] -= 1
            counts[A] += 1
        # Rule 5: X + Y -> Z
        elif counts[X] >= 1 and counts[Y] >= 1:
            counts[X] -= 1
            counts[Y] -= 1
            counts[Z] += 1

        return counts

    def _get_answer(self, A_machine: int, B_machine: int, C_machine: int) -> list[list[int]]:
        """Calculate the answer for a given input"""
        # counts for A B C X Y Z
        counts = [A_machine, B_machine, C_machine, 0, 0, 0]
        states = [counts]

        for _ in range(self.config.max_iterations):
            new_counts = self._apply_rule(counts[:])
            if new_counts in states:
                break
            states.append(new_counts)
            counts = new_counts

        return states

    def __getitem__(self, idx: int) -> dict:
        """Generate a single String Splitting question"""
        rng = Random(self.seed + idx)

        A_machine = rng.randint(self.config.min_initial_machines, self.config.max_initial_machines)
        B_machine = rng.randint(self.config.min_initial_machines, self.config.max_initial_machines)
        C_machine = rng.randint(self.config.min_initial_machines, self.config.max_initial_machines)

        states = self._get_answer(A_machine, B_machine, C_machine)
        answer = states[-1]
        answer_str = " ".join(str(x) for x in answer)

        return {
            "question": QUESTION_TEMPLATE.format(A_machine=A_machine, B_machine=B_machine, C_machine=C_machine),
            "answer": answer_str,
            "metadata": {"states": states, "solution": answer},
        }


register_dataset("string_splitting", StringSplittingDataset, StringSplittingConfig)

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset


def is_prime(n):
    """Return True if n is a prime number, False otherwise."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def is_composite(n):
    """
    Return True if n is composite.
    (Composite means an integer greater than 1 that is not prime.)
    """
    return n > 1 and not is_prime(n)


def generate_dynamic_puzzle(difficulty, rng):
    """
    Dynamically generates a 7-statement self-referential puzzle.

    The seven statements (with parameters determined by this function) are:

      1. "At least a of these 7 statements are true."
      2. "At most b of these 7 statements are false."
      3. "Exactly c of these 7 statements are true."
      4. "Exactly d of these 7 statements are false."
      5. "Either Statement 3 or Statement 4 is true, but not both."
      6. "The number of true statements is a prime number."
      7. "The number of false statements is a composite number."

    The idea is to choose an intended number T (1 ≤ T ≤ 6) of true statements
    and then “plant” an intended solution. In our construction the truth values
    for Statements 6 and 7 are forced by T (e.g. Statement 6 should be true exactly
    when T is prime). For the first four statements the numeric parameters (a, b, c, d)
    are chosen so that the statement evaluates correctly when compared to T.

    The difficulty parameter (an integer, e.g. 1 for easy up to 10 for hard)
    influences how “borderline” the numeric choices are. At lower difficulty the numbers
    are chosen with a clear gap; at higher difficulty they are chosen closer to T.

    Returns:
        dict: A puzzle dictionary containing:
              - 'n': number of statements (always 7 here),
              - 'statements_text': a list of 7 strings (one per statement),
              - 'parameters': a dict with the numeric parameters (for statements 1-4),
              - 'intended_assignment': the intended truth values (list of 7 booleans),
              - 'intended_T': the intended number of true statements.
    """
    n = 7

    # Choose an intended number of true statements, T, from 1 to 6 (nontrivial).
    T = rng.choice(range(1, n))

    # For the global statements (6 and 7), the intended truth is forced:
    intended6 = is_prime(T)  # Statement 6 must be true if T is prime.
    intended7 = is_composite(n - T)  # Statement 7 must be true if (# false) is composite.

    # Among statements 1-5, we need exactly k trues such that overall the total becomes T.
    # Let k = T - (truth from statements 6 and 7).
    forced_true_count = (1 if intended6 else 0) + (1 if intended7 else 0)
    k = T - forced_true_count
    # k must be between 0 and 5.
    if not (0 <= k <= 5):
        # If for some reason it is not in range, fall back to a known configuration (T=4).
        T = 4
        intended6 = False
        intended7 = False
        k = 4  # so that overall T=4.
        intended_assignment_15 = [True, True, True, True, False]
    else:
        # For statements 1-5, randomly choose which ones are intended true.
        # We'll index these as 0..4 corresponding to statements 1..5.
        intended_assignment_15 = [False] * 5
        if k > 0:
            true_indices = set(rng.sample(range(5), k))
            for i in true_indices:
                intended_assignment_15[i] = True

    # Now, for statements 1-4, choose numeric parameters based on whether the statement is
    # intended to be true or false. We use the difficulty parameter to control the "margin."
    #
    # For statement 1: "At least a of these 7 statements are true."
    # The condition is: T >= a.
    def choose_at_least_param(T, intended, diff, rng):
        # diff will be used as a margin factor: lower diff => wider gap.
        if intended:  # must have a <= T.
            # At easy difficulty, choose a clearly below T (if possible).
            low = 1
            high = T
            # At lower difficulty, bias toward the lower end.
            return rng.randint(low, high)
        else:  # must have a > T.
            low = T + 1
            high = n  # a can be at most n.
            if low > high:
                return n
            return rng.randint(low, high)

    a_param = choose_at_least_param(T, intended_assignment_15[0], difficulty, rng)

    # For statement 2: "At most b of these 7 statements are false."
    # F = n - T, so condition is: (n - T) <= b   <=>   T >= n - b.
    def choose_at_most_param(T, intended, diff, rng):
        if intended:  # b must be >= n - T.
            low = n - T
            high = n
            return rng.randint(low, high)
        else:
            # b must be < n - T.
            low = 0
            high = max(n - T - 1, 0)
            return rng.randint(low, high)

    b_param = choose_at_most_param(T, intended_assignment_15[1], difficulty, rng)

    # For statement 3: "Exactly c of these 7 statements are true."
    def choose_exactly_true_param(T, intended, diff, rng):
        if intended:
            return T
        else:
            choices = [x for x in range(0, n + 1) if x != T]
            return rng.choice(choices)

    c_param = choose_exactly_true_param(T, intended_assignment_15[2], difficulty, rng)

    # For statement 4: "Exactly d of these 7 statements are false."
    # Condition: (n - T) == d.
    def choose_exactly_false_param(T, intended, diff, rng):
        false_count = n - T
        if intended:
            return false_count
        else:
            choices = [x for x in range(0, n + 1) if x != false_count]
            return rng.choice(choices)

    d_param = choose_exactly_false_param(T, intended_assignment_15[3], difficulty, rng)

    # For statement 5: "Either Statement 3 or Statement 4 is true, but not both."
    # We do not need a parameter here; the intended condition is that the truth values for
    # statements 3 and 4 (which are positions 2 and 3 in our 0-indexed list) differ.
    # The intended truth for statement 5 is taken from our assignment.
    # (Later the verification function will check: solution[2] != solution[3].)

    # Build the intended assignment for all 7 statements.
    # For statements 1-5, we use our generated intended_assignment_15.
    intended_assignment = [
        intended_assignment_15[0],
        intended_assignment_15[1],
        intended_assignment_15[2],
        intended_assignment_15[3],
        intended_assignment_15[4],
        intended6,
        intended7,
    ]

    # (If the total intended true count doesn't equal T, adjust statement 5.)
    current_T = sum(intended_assignment)
    if current_T != T:
        # Since only statement 5 is free (its parameter wasn't numeric),
        # force its intended truth to be what is needed.
        intended_assignment[4] = T - (current_T - (1 if intended_assignment[4] else 0)) == 1

    # Now build the text for each statement.
    statements_text = [
        f"Statement 1: 'At least {a_param} of these 7 statements are true.'",
        f"Statement 2: 'At most {b_param} of these 7 statements are false.'",
        f"Statement 3: 'Exactly {c_param} of these 7 statements are true.'",
        f"Statement 4: 'Exactly {d_param} of these 7 statements are false.'",
        "Statement 5: 'Either Statement 3 or Statement 4 is true, but not both.'",
        "Statement 6: 'The number of true statements is a prime number.'",
        "Statement 7: 'The number of false statements is a composite number.'",
    ]

    return {
        "n": n,
        "statements_text": statements_text,
        "parameters": {
            "a": a_param,
            "b": b_param,
            "c": c_param,
            "d": d_param,
        },
        "intended_assignment": intended_assignment,
        "intended_T": T,
        "difficulty": difficulty,
    }


def verify_solution_dynamic(puzzle, solution):
    """
    Verifies a candidate solution for a dynamically generated puzzle.

    The rules are:
      - If a statement is marked True, then its claim must hold.
      - If a statement is marked False, then its claim must fail.

    The conditions are as follows:
      1. "At least a of these 7 statements are true."  => (T >= a)
      2. "At most b of these 7 statements are false."   => (F <= b)
      3. "Exactly c of these 7 statements are true."    => (T == c)
      4. "Exactly d of these 7 statements are false."   => (F == d)
      5. "Either Statement 3 or Statement 4 is true, but not both." => (solution[2] != solution[3])
      6. "The number of true statements is a prime number." => is_prime(T)
      7. "The number of false statements is a composite number." => is_composite(F)

    Parameters:
       puzzle (dict): The puzzle dictionary returned by generate_dynamic_puzzle.
       solution (list of bool): A candidate assignment (length 7).

    Returns:
       bool: True if candidate is self-consistent; False otherwise.
    """
    n = puzzle["n"]
    if len(solution) != n:
        return False
    T = sum(solution)
    F = n - T
    params = puzzle["parameters"]

    # Statement 1: "At least a of these 7 statements are true."
    cond1 = T >= params["a"]
    if solution[0] and not cond1:
        return False
    if not solution[0] and cond1:
        return False

    # Statement 2: "At most b of these 7 statements are false."
    cond2 = F <= params["b"]
    if solution[1] and not cond2:
        return False
    if not solution[1] and cond2:
        return False

    # Statement 3: "Exactly c of these 7 statements are true."
    cond3 = T == params["c"]
    if solution[2] and not cond3:
        return False
    if not solution[2] and cond3:
        return False

    # Statement 4: "Exactly d of these 7 statements are false."
    cond4 = F == params["d"]
    if solution[3] and not cond4:
        return False
    if not solution[3] and cond4:
        return False

    # Statement 5: "Either Statement 3 or Statement 4 is true, but not both."
    cond5 = solution[2] != solution[3]
    if solution[4] and not cond5:
        return False
    if not solution[4] and cond5:
        return False

    # Statement 6: "The number of true statements is a prime number."
    cond6 = is_prime(T)
    if solution[5] and not cond6:
        return False
    if not solution[5] and cond6:
        return False

    # Statement 7: "The number of false statements is a composite number."
    cond7 = is_composite(F)
    if solution[6] and not cond7:
        return False
    if not solution[6] and cond7:
        return False

    return True


def print_puzzle_dynamic(puzzle):
    """Prints the dynamically generated puzzle."""
    x = ""
    for stmt in puzzle["statements_text"]:
        x = x + " - " + stmt + "\n"
    return x


def solve_puzzle_dynamic(puzzle):
    """
    Searches all 2^7 possible truth assignments and returns those that
    are self-consistent with the generated puzzle.
    """
    n = puzzle["n"]
    valid_solutions = []
    for i in range(2**n):
        candidate = [(i >> j) & 1 == 1 for j in range(n)]
        if verify_solution_dynamic(puzzle, candidate):
            valid_solutions.append(candidate)
    return valid_solutions


@dataclass
class SelfReferenceConfig:
    """Configuration for SelfReference puzzle generation"""

    difficulty: int = 5
    seed: Optional[int] = None
    size: int = 500

    def validate(self):
        """Validate configuration parameters"""
        assert 1 <= self.difficulty <= 10, "difficulty must be between 1 and 10"


class SelfReferenceDataset(ProceduralDataset):
    """Generates self-referential puzzles"""

    def __init__(self, config: SelfReferenceConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single SelfReference task

        Returns:
            dict with keys:
                - question: str, the task description
                - answer: str, a solution string
                - metadata: dict with generation parameters
        """
        rng = Random(self.seed + idx)

        # Generate puzzle
        puzzle = generate_dynamic_puzzle(self.config.difficulty, rng)
        puzz_s = (
            "Given the truthfulness of these statements, please tell me the number of possible solutions: \n"
            + print_puzzle_dynamic(puzzle)
        )

        # Solve puzzle
        solutions = solve_puzzle_dynamic(puzzle)
        for idx, sol in enumerate(solutions, start=1):
            sol_str = ["True" if s else "False" for s in sol]
        answer = len(solutions)

        return {
            "question": puzz_s,
            "answer": answer,
            "metadata": {},
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the SelfReference task.

        The function awards 1.0 for a correct answer.

        Args:
            answer (Optional[str]): The user's answer.
            entry (dict[str, Any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        if answer == None:
            return 0.0
        if str(answer) != str(entry["answer"]):
            return 0.1
        else:
            return 1.0  # Yay


register_dataset("self_reference", SelfReferenceDataset, SelfReferenceConfig)

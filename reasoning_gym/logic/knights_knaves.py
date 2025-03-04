import copy
import itertools
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

import numpy as np

from reasoning_gym.factory import ProceduralDataset, register_dataset

COMMON_NAMES = [
    "Emma",
    "Liam",
    "Olivia",
    "Noah",
    "Ava",
    "Ethan",
    "Sophia",
    "Mason",
    "Isabella",
    "William",
    "Mia",
    "James",
    "Charlotte",
    "Benjamin",
    "Amelia",
    "Lucas",
    "Harper",
    "Henry",
    "Evelyn",
    "Alexander",
    "Abigail",
    "Michael",
    "Emily",
    "Daniel",
    "Elizabeth",
    "Jacob",
    "Sofia",
    "Logan",
    "Avery",
    "Jackson",
    "Ella",
    "Sebastian",
    "Scarlett",
    "Jack",
    "Grace",
    "Aiden",
    "Chloe",
    "Owen",
    "Victoria",
    "Samuel",
    "Riley",
    "Matthew",
    "Aria",
    "Joseph",
    "Lily",
    "Luke",
    "Aurora",
    "David",
    "Zoey",
    "Oliver",
    "Penelope",
]

KNIGHT_KNAVE_PAIRS = [
    ["a knight", "a knave"],
    ["a pioneer", "a laggard"],
    ["a saint", "a sinner"],
    ["a hero", "a villain"],
    ["an angel", "a devil"],
    ["an altruist", "an egoist"],
    ["a sage", "a fool"],
]

VALID_ROLES = {pair[i].split()[1] for pair in KNIGHT_KNAVE_PAIRS for i in range(2)}

PREFIX = (
    "A very special island is inhabited only by {knight}s and {knave}s. "
    + "{Knight}s always tell the truth, and {knave}s always lie. "
)

POSTFIX = "So who is {a_knight} and who is {a_knave}?"

TEMPLATES = [
    "{name} said that {content}.",
    "{name} told you that {content}.",
    '{name} said, "{content}."',
    '{name} stated, "{content}".',
    'According to {name}, "{content}".',
    'In {name}\'s words: "{content}".',
    '{name} remarked, "{content}".',
    '"{content}," {name} declared.',
    '{name} was heard saying, "{content}".',
    "{name} expressed that {content}.",
    '"{content}" - {name}.',
    'As {name} put it, "{content}".',
    '{name} asserted: "{content}".',
    '"{content}," {name} mentioned.',
    '{name} commented, "{content}".',
    'In a statement by {name}: "{content}".',
    '{name} noted, "{content}".',
    '"{content}," {name} claimed.',
]


@dataclass
class KnightsKnavesConfig:
    """
    Configuration for knights and knaves task generation.

    :param n_people: Number of people in the problem
    :param depth_constraint: Maximum depth of each person's statement
    :param width_constraint: Maximum width (number of branches) of each person's statement
    :param size: Virtual size of dataset
    :param seed: Random seed
    """

    n_people: int = 2
    depth_constraint: int = 2
    width_constraint: int = 2
    size: int = 500
    seed: Optional[int] = None

    def validate(self):
        assert 1 <= self.n_people, "Number of people must be at least 1"
        assert 1 <= self.depth_constraint, "Depth constraint must be at least 1"
        assert 1 <= self.width_constraint, "Width constraint must be at least 1"


class KKProblemSampler:
    def __init__(self, rand_seed: int, n_people: int, depth_constraint: int = 2, width_constraint: int = 2):
        self.rng = np.random.default_rng(rand_seed)
        self.n_people = n_people
        self.depth_constraint = depth_constraint
        self.width_constraint = width_constraint

    def sample(self):
        statements = tuple(
            self._sample_statement(person_id, self.depth_constraint) for person_id in range(self.n_people)
        )
        return self._immutable_statements(statements)

    def sample_valid_problems(
        self,
        n_problems: int,
        max_retry: int = 1000,
        skip_no_solution: bool = True,
        skip_multiple_solutions: bool = True,
    ):
        problems = []
        unique_statements = set()
        for _ in range(n_problems):
            for _ in range(max_retry):
                statements = self.sample()
                if statements in unique_statements:
                    continue
                solutions = KnightsKnavesDataset.find_solution(statements)
                if len(solutions) == 0 and skip_no_solution:
                    continue
                if len(solutions) > 1 and skip_multiple_solutions:
                    continue
                sol = solutions[0] if len(solutions) > 0 else None
                problems.append({"statements": statements, "solution": sol, "all_solutions": solutions})
                unique_statements.add(statements)
                break
        return problems

    def _sample_statement(self, person_id: int, depth_constraint: int):
        dice = self.rng.integers(0, 6)
        if depth_constraint == 1 or dice == 0:
            while True:
                knight_or_knave = self.rng.choice(["telling-truth", "lying"])
                person = self.rng.integers(0, self.n_people)
                # prevent the contradiction "I am lying"
                if not (knight_or_knave == "lying" and person == person_id):
                    return (knight_or_knave, person)
        if dice == 1:
            return ("not", self._sample_statement(person_id, depth_constraint - 1))
        if dice in [2, 3]:
            operator = ["and", "or"][dice - 2]
            n_substatements = self.rng.integers(2, self.width_constraint + 1)
            return (operator,) + self._sample_substatements(person_id, depth_constraint, n_substatements)
        if dice in [4, 5]:
            operator = ["->", "<=>"][dice - 4]
            return (operator,) + self._sample_substatements(person_id, depth_constraint, 2)

    def _sample_substatements(self, person_id: int, depth_constraint: int, count: int, dedup: bool = True):
        sub_statements = []
        dedup_set = set()
        while True:
            stmt = self._sample_statement(person_id, depth_constraint - 1)
            if dedup:
                if stmt in dedup_set:
                    continue
                dedup_set.add(stmt)
            sub_statements.append(stmt)
            if len(sub_statements) == count:
                break
        return tuple(sub_statements)

    def _immutable_statements(self, mutable_statements):
        def _make_immutable(x):
            if isinstance(x, (list, tuple)):
                return tuple(_make_immutable(child) for child in x)
            if isinstance(x, np.str_):
                return str(x)
            if isinstance(x, np.int64):
                return int(x)
            return x

        return tuple(_make_immutable(s) for s in mutable_statements)


class KKProblemFormatter:
    def __init__(self, rand_seed, problem):
        self.rng = np.random.default_rng(rand_seed)
        self.problem = problem

    def format_problem(self):
        statements = copy.deepcopy(self.problem["statements"])
        n_people = len(statements)
        names = list(self.rng.choice(COMMON_NAMES, size=n_people, replace=False))
        knight_knave = self.rng.choice(KNIGHT_KNAVE_PAIRS)
        knight_knave = {
            "knight": knight_knave[0].split()[1],
            "knave": knight_knave[1].split()[1],
            "a_knight": knight_knave[0],
            "a_knave": knight_knave[1],
        }
        knight_knave["Knight"] = knight_knave["knight"].capitalize()
        knight_knave["Knave"] = knight_knave["knave"].capitalize()

        text = PREFIX.format(**knight_knave)
        text += f"You meet {n_people} inhabitants: "
        text += ", ".join(names[:-1]) + ", and " + names[-1] + "."

        text_statements = []
        for i, stmt in enumerate(statements):
            tmpl = self.rng.choice(TEMPLATES)
            content = self._format_statement(names, knight_knave, stmt)
            text_statements.append(" " + tmpl.format(name=names[i], content=content))

        text += "".join(text_statements)
        text += " " + POSTFIX.format(**knight_knave)
        format = ", ".join(f"{name} is a {knight_knave['knight']}/{knight_knave['knave']}" for name in names[:-1])
        if len(names) > 1:
            format += f", and {names[-1]} is a {knight_knave['knight']}/{knight_knave['knave']}"
        else:
            format = f"{names[0]} is a {knight_knave['knight']}/{knight_knave['knave']}"

        text += f' (Format your answer like: "{format}")'

        if self.problem["solution"] is None:
            solution_text = "No valid solution exists."
        else:
            solution_stmts = []
            for name, indicator in zip(names, self.problem["solution"]):
                if indicator:
                    solution_stmts.append(name + " is " + knight_knave["a_knight"])
                else:
                    solution_stmts.append(name + " is " + knight_knave["a_knave"])
            solution_text = ", ".join(solution_stmts[:-1]) + ", and " + solution_stmts[-1] + "."
        return {
            "quiz": text,
            "names": names,
            "knight_knave": knight_knave,
            "solution": self.problem["solution"],
            "solution_text": solution_text,
        }

    def _format_statement(self, names, knight_knave, statement, depth=0):
        """
        Recursively format a logical statement with appropriate parentheses based on depth.

        Args:
            names: List of people's names
            knight_knave: Dictionary with knight/knave terminology
            statement: Logical statement tuple to format
            depth: Current nesting depth (0 = top level)
        """
        # Base case: this is a primitive statement
        if statement[0] in ("telling-truth", "lying"):
            return self._format_knight_knave(names, knight_knave, statement)

        # Handle negation
        if statement[0] == "not":

            # Special case: If negating a primitive statement, use the complementary term directly
            if statement[1][0] in ("telling-truth", "lying"):
                # Map "telling-truth" to "lying" and vice versa
                complementary_statement = (
                    "lying" if statement[1][0] == "telling-truth" else "telling-truth",
                    statement[1][1],
                )
                return self._format_knight_knave(names, knight_knave, complementary_statement)
            else:
                # For complex statements, use the verbose "it is not the case that" format
                inner_content = self._format_statement(names, knight_knave, statement[1], depth + 1)
                if statement[1][0] not in ("telling-truth", "lying"):
                    inner_content = f"({inner_content})"
                return f"it is not the case that {inner_content}"

        # Handle AND/OR
        if statement[0] in ["and", "or"]:
            formatted_substmts = []
            for sub_stmt in statement[1:]:
                sub_content = self._format_statement(names, knight_knave, sub_stmt, depth + 1)
                # Always add parentheses for complex subexpressions in AND/OR
                if sub_stmt[0] not in ("telling-truth", "lying"):
                    sub_content = f"({sub_content})"
                formatted_substmts.append(sub_content)
            connector = f" {statement[0]} "
            return connector.join(formatted_substmts)

        # Handle implication
        if statement[0] == "->":
            antecedent = self._format_statement(names, knight_knave, statement[1], depth + 1)
            consequent = self._format_statement(names, knight_knave, statement[2], depth + 1)

            # Always add parentheses for complex expressions in implications
            if statement[1][0] not in ("telling-truth", "lying"):
                antecedent = f"({antecedent})"
            if statement[2][0] not in ("telling-truth", "lying"):
                consequent = f"({consequent})"

            return f"if {antecedent} then {consequent}"

        # Handle biconditional
        if statement[0] == "<=>":
            left = self._format_statement(names, knight_knave, statement[1], depth + 1)
            right = self._format_statement(names, knight_knave, statement[2], depth + 1)

            # Always add parentheses for complex expressions in biconditionals
            if statement[1][0] not in ("telling-truth", "lying"):
                left = f"({left})"
            if statement[2][0] not in ("telling-truth", "lying"):
                right = f"({right})"

            return f"{left} if and only if {right}"

        # This should not happen with well-formed statements
        raise ValueError(f"Unknown statement type: {statement[0]}")

    def _format_knight_knave(self, names, knight_knave, statement, negation=False):
        assert statement[0] in ("telling-truth", "lying")
        text = names[statement[1]] + " is "
        if negation:
            text += "not "
        text += {"telling-truth": knight_knave["a_knight"], "lying": knight_knave["a_knave"]}[statement[0]]
        return text


class KnightsKnavesDataset(ProceduralDataset):
    """
    Generates random knights and knaves problems.

    This implementation is adapted from the Knights and Knaves problem generator in:
    https://github.com/AlphaPav/mem-kk-logic

    As described in the paper:
    @article{xie2024memorization,
    title={On Memorization of Large Language Models in Logical Reasoning},
    author={Chulin Xie and Yangsibo Huang and Chiyuan Zhang and Da Yu and Xinyun Chen and Bill Yuchen Lin and Bo Li and Badih Ghazi and Ravi Kumar},
    year={2024},
    eprint={2410.23123},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2410.23123},
    }
    """

    def __init__(self, config: KnightsKnavesConfig):
        super().__init__(config, seed=config.seed, size=config.size)

    @staticmethod
    def find_solution(statements):
        """Find solutions given a list of statements."""
        n_people = len(statements)
        single_statement = ("and",) + tuple(
            ("<=>", ("telling-truth", i), statements[i]) for i in range(len(statements))
        )
        # Brute force
        solutions = []
        for assignments in itertools.product([True, False], repeat=n_people):
            # if KnightsKnavesDataset.test_satisfiability(single_statement, assignments):
            if KnightsKnavesDataset.test_satisfiability(single_statement, assignments):
                solutions.append(assignments)
        return solutions

    @staticmethod
    def test_satisfiability(statement, assignments):
        """Recursively test if a statement is satisfied under given assignments."""
        if statement[0] == "telling-truth":
            return assignments[statement[1]]
        if statement[0] == "lying":
            return not assignments[statement[1]]
        if statement[0] == "not":
            return not KnightsKnavesDataset.test_satisfiability(statement[1], assignments)
        if statement[0] == "and":
            return np.all(
                [KnightsKnavesDataset.test_satisfiability(statement[i], assignments) for i in range(1, len(statement))]
            )
        if statement[0] == "or":
            return np.any(
                [KnightsKnavesDataset.test_satisfiability(statement[i], assignments) for i in range(1, len(statement))]
            )
        if statement[0] == "->":
            val1 = KnightsKnavesDataset.test_satisfiability(statement[1], assignments)
            val2 = KnightsKnavesDataset.test_satisfiability(statement[2], assignments)
            return (not val1) or val2
        if statement[0] == "<=>":
            val1 = KnightsKnavesDataset.test_satisfiability(statement[1], assignments)
            val2 = KnightsKnavesDataset.test_satisfiability(statement[2], assignments)
            return (val1 and val2) or ((not val1) and (not val2))
        raise KeyError(f"Unknown statement: {statement}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Generate a single knights and knaves problem item.

        Args:
            idx: Index of the item to generate

        Returns:
            dict containing at least:
                - question: str (the puzzle in natural language)
                - answer: str (the solution in text)
                - metadata: dict (additional problem details)
        """
        rng = Random(self.seed + idx if self.seed is not None else None)
        return self.__generate_problem(rng)

    def __generate_problem(self, rng: Random) -> dict[str, Any]:
        """
        Generate a single knights and knaves problem with a unique solution.
        """

        # Sample a valid problem using the original KKProblemSampler logic
        # Use the sampler to generate a valid problem
        sampler = KKProblemSampler(
            rand_seed=rng.randint(0, 2**32),
            n_people=self.config.n_people,
            depth_constraint=self.config.depth_constraint,
            width_constraint=self.config.width_constraint,
        )
        problems = sampler.sample_valid_problems(1, skip_no_solution=True, skip_multiple_solutions=True)
        problem = problems[0]

        # Format the problem
        formatter = KKProblemFormatter(rand_seed=rng.randint(0, 2**32), problem=problem)
        formatted = formatter.format_problem()

        # Prepare the return dictionary
        question = formatted["quiz"]
        answer = formatted["solution_text"]
        metadata = {
            "statements": problem["statements"],
            "solution": problem["solution"],
            "names": formatted["names"],
            "knight_knave_terms": formatted["knight_knave"],
        }

        return {"question": question, "answer": answer, "metadata": metadata}

    @staticmethod
    def _normalize_answer(answer: str) -> set[tuple[str, str]]:
        """Convert answer string into normalized set of (name, role) tuples"""
        # Remove common punctuation and standardize spacing
        answer = answer.lower().strip().replace(".", " ").replace(",", " ").replace(")", " ").replace("(", " ")

        # Split on 'and' or spaces for different formats
        parts = [p.strip() for p in answer.replace(" and ", " ").split()]

        # Extract name-role pairs
        assignments = set()
        current_name = None

        for part in parts:
            if part in ["is", "a", "an"]:
                continue
            if part in VALID_ROLES:
                if current_name:
                    assignments.add((current_name, part))
                    current_name = None
            else:
                current_name = part

        return assignments

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Score an answer against the oracle answer."""
        if not isinstance(answer, str) or len(answer) == 0:
            return 0.0

        try:
            oracle_assignments = self._normalize_answer(entry["answer"])
            answer_assignments = self._normalize_answer(answer)

            # Full credit for exact assignments regardless of order
            if oracle_assignments == answer_assignments:
                return 1.0

            # Partial credit if all names are present and some assignments match
            if len(oracle_assignments) == len(answer_assignments):
                matching = len(oracle_assignments.intersection(answer_assignments))
                if matching > 0:
                    return 0.3 + (0.7 * matching / len(oracle_assignments))

        except Exception:
            pass
        return 0.0


register_dataset("knights_knaves", KnightsKnavesDataset, KnightsKnavesConfig)

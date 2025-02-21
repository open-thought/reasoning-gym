"""
puzzle_generator.py

This is a driver script that can be used to generate new zebra puzzles.
"""

from collections import OrderedDict
from itertools import product
from random import Random
from typing import Iterable, Type

from tabulate import tabulate

from reasoning_gym.logic.contrib.logic_puzzle.literals import *
from reasoning_gym.logic.contrib.logic_puzzle.puzzle import Puzzle
from reasoning_gym.logic.contrib.logic_puzzle.sat_utils import itersolve

from .clues import Clue, beside, consecutive, found_at, left_of, not_at, one_between, right_of, same_house, two_between


def generate_found_at(puzzle: Puzzle, solution: OrderedDict[Literal, int]) -> set[Clue]:
    """Generate the `found_at` / `not_at` Clue instances"""
    clues: set[Clue] = set()
    for element, loc in solution.items():
        clues.add(found_at(element, loc))

    return clues


def generate_not_found_at(puzzle: Puzzle, solution: dict[Literal, int]) -> set[Clue]:
    """Generate the `found_at` / `not_at` Clue instances"""
    clues: set[Clue] = set()
    for element, loc in solution.items():
        for house in puzzle.houses:
            if house != loc:
                clues.add(not_at(element, house))

    return clues


def generate_same_house(puzzle: Puzzle, solution: OrderedDict[Literal, int]) -> set[Clue]:
    """Generate the `same_house` Clue instances"""

    clues: set[Clue] = set()
    for house in puzzle.houses:
        items_at_house = {item: loc for item, loc in solution.items() if loc == house}
        pairs: set[tuple[Literal, Literal]] = {
            (item1, item2) for item1, item2 in product(items_at_house, repeat=2) if item1 != item2
        }
        for pair in pairs:
            clues.add(same_house(pair[0], pair[1], puzzle.houses))

    return clues


def generate_consecutive_beside(puzzle: Puzzle, solution: OrderedDict[Literal, int]) -> set[Clue]:
    """Generate the `consecutive` / `beside` Clue instances

    (Note that consecutive is just a more informative version of beside. Since they have the same
    structure, for every possible combination we'll just keep one.
    """

    clues: set[Clue] = set()
    for left, right in zip(puzzle.houses, puzzle.houses[1:]):
        items_left = {item: loc for item, loc in solution.items() if loc == left}
        items_right = {item: loc for item, loc in solution.items() if loc == right}
        pairs: set[tuple[Literal, Literal]] = {(item1, item2) for item1, item2 in product(items_left, items_right)}
        # sorted, no hash randomization
        for pair in sorted(pairs):
            # consecutive is just a more informative version of beside, but they have same structure
            # because of this, don't include both
            if puzzle.rng.randint(0, 1) == 0:
                clues.add(consecutive(pair[0], pair[1], puzzle.houses))
            else:
                clues.add(beside(pair[0], pair[1], puzzle.houses))

    return clues


def generate_left_right_of(puzzle: Puzzle, solution: dict[Literal, int]) -> set[Clue]:
    """Generate the `left_of` / `right_of` Clue instances
    Note that since (x left-of y) is guaranteed to be redundant with (b right-of a), we only add
    one of these clues to the final set.
    """

    clues: set[Clue] = set()
    for left, right in product(puzzle.houses, puzzle.houses):
        if left >= right:
            continue

        items_left = {item: loc for item, loc in solution.items() if loc == left}
        items_right = {item: loc for item, loc in solution.items() if loc == right}
        pairs: set[tuple[Literal, Literal]] = {(item1, item2) for item1, item2 in product(items_left, items_right)}
        # sorted, no hash randomization
        for pair in sorted(pairs):
            if puzzle.rng.randint(0, 1) == 0:
                clues.add(left_of(pair[0], pair[1], puzzle.houses))
            else:
                clues.add(right_of(pair[1], pair[0], puzzle.houses))

    return clues


def generate_one_between(puzzle: Puzzle, solution: dict[Literal, int]) -> set[Clue]:
    """Generate the `one_between` Clue instances"""

    clues: set[Clue] = set()
    for left, right in zip(puzzle.houses, puzzle.houses[2:]):
        items_left = {item: loc for item, loc in solution.items() if loc == left}
        items_right = {item: loc for item, loc in solution.items() if loc == right}
        pairs: set[tuple[Literal, Literal]] = {(item1, item2) for item1, item2 in product(items_left, items_right)}
        for pair in pairs:
            clues.add(one_between(pair[0], pair[1], puzzle.houses))

    return clues


def generate_two_between(puzzle: Puzzle, solution: dict[Literal, int]) -> set[Clue]:
    """Generate the `two_between` Clue instances"""

    clues: set[Clue] = set()
    for left, right in zip(puzzle.houses, puzzle.houses[3:]):
        items_left = {item: loc for item, loc in solution.items() if loc == left}
        items_right = {item: loc for item, loc in solution.items() if loc == right}
        pairs: set[tuple[Literal, Literal]] = {(item1, item2) for item1, item2 in product(items_left, items_right)}
        for pair in pairs:
            clues.add(two_between(pair[0], pair[1], puzzle.houses))

    return clues


def has_unique_solution(puzzle: Puzzle, clues: Iterable[Clue]) -> bool:
    """Test if a puzzle has a unique solution under a given set of clues."""

    with puzzle.with_clues(clues):
        # print(f"Testing puzzle with {len(puzzle.clues)} clues")
        solutions = itersolve(puzzle.as_cnf())
        _first_solution = next(solutions)

        # test if second solution exists or not; if it doesn't, uniquely solvable
        if next(solutions, None) is None:
            return True
        else:
            return False


def try_to_remove(puzzle: Puzzle, clues: set[Clue], n: int, must_have=set()) -> set[Clue]:
    """
    Attempt to remove n clues from a set of candidate clues; if we are able to, return the new,
    smaller set of clues. If not, return the original set.
    """

    def weight(clue: Clue) -> float:
        # relative probabilities of each type of clue being selected for removal
        weights: dict[Type[Clue], float] = {
            not_at: 0.75,
            found_at: 0.75,
            same_house: 0.75,
            beside: 1.2,
            left_of: 1.2,
            right_of: 1.2,
            one_between: 1.5,
            two_between: 1.5,
        }

        return weights.get(type(clue), 1)

    # sorted, no hash randomization
    weights = [weight(clue) for clue in sorted(clues)]
    candidates: set[Clue] = set(puzzle.rng.choices(sorted(clues), weights, k=n))
    candidates = candidates - must_have
    clues = clues.difference(candidates)
    if has_unique_solution(puzzle, clues):
        # print(f"Removed {len(candidates)} clues.")
        return clues

    # we needed at least one of those, add them all back
    clues = clues | candidates
    return clues


def reduce_individually(
    puzzle: Puzzle, clues: set[Clue], removed: set[Clue], must_have=set()
) -> tuple[set[Clue], set[Clue]]:
    """
    Attempt to remove each candidate clue one by one.

    The sets `clues` and `removed` are modified in-place. Unnecessary clues get removed from `clues`
    and added to `removed`. If no clues can be removed, we return the original two sets.
    """

    candidates = puzzle.rng.sample(sorted(clues), len(clues))
    for clue in candidates:
        if clue not in must_have:
            clues.remove(clue)
            if has_unique_solution(puzzle, clues):
                # print(f"Removed {clue=}")
                removed.add(clue)
                continue  # we were fine to remove this clue
        clues.add(clue)

    return clues, removed


def reduce_clues(puzzle: Puzzle, clues: set[Clue], must_have=set()) -> tuple[set[Clue], set[Clue]]:
    """
    Reduce a set of clues to a minimally solvable set.

    A minimally solvable 5-house, 4-attribute puzzle takes between 10 and 20 clues to solve. The
    original set of clues will likely be in the hundreds, and the majority are likely to be
    redundant. We can quickly reduce the set of clues by batch removing clues from the large
    candidate pool.

    The algorithm for batch reduction:
     1. shuffle all the clues
     2. attempt to remove 10% of the clues; with this 90%-clue set, test if the puzzle is solvable.
     3a. if yes: keep them removed, go back to 2 and repeat
     3b. if no: add them back, keep going to 4
     4. the same as step (3), but this time trying to remove 5% of the clues
     5. the same as step (3), but this time trying to remove a single clue

    After we've tried and failed to remove a *single* clue, then the (first part of the) reduction
    algorithm is done; having that clue was necessary for us to have a unique solution. This doesn't
    necessarily mean that *all* the clues are need, though, which is what the secondary reduction
    is for.

    The *secondary reduction process* is much simpler: now that the set is narrowed substantially,
    we can be more brute-forcey. Attempt to remove each clue and see if the puzzle is still
    solvable.

    However, the secondary reduction process can result in a puzzle that is *too hard* to solve
    (though technically uniquely solvable by a computer or sufficiently skilled human). This
    algorithm returns a second set of clues that were removed during the secondary reduction
    process. These can be thought of as extra clues to add or hints to give to anyone solving the
    puzzle.

    """

    # this is a stupid way to shuffle the set of clues without modifying it
    minimal_clues = set(puzzle.rng.sample(list(clues), k=len(clues)))
    while True:
        # print(f"There are {len(minimal_clues)} clues in ba sing se")

        # Walrus time!
        #
        # If the size of minimal_clues before we try to remove some clues is greater than the size
        # after, then those clues were fine to remove. Go back to the top of the loop and keep
        # removing more. But if the size is the same, we needed some of those clues. Try to remove
        # a smaller amount.
        #
        # We use the walrus operator to update minimal_clues in place during the comparison. It'll
        # either be a reduced set of clues or the original set of clues.
        if len(minimal_clues) > len(
            (minimal_clues := try_to_remove(puzzle, minimal_clues, len(minimal_clues) // 10, must_have))
        ):
            continue

        if len(minimal_clues) != len(
            (minimal_clues := try_to_remove(puzzle, minimal_clues, len(minimal_clues) // 20, must_have))
        ):
            continue

        if len(minimal_clues) == len((minimal_clues := try_to_remove(puzzle, minimal_clues, 1, must_have))):
            break

    # secondary reduction time! While we can still remove clues, do so; then we're done.
    # print(f"Starting the secondary reduction.")
    removed_clues: set[Clue] = set()
    while True:
        minimal_clues_size = len(minimal_clues)
        minimal_clues, removed_clues = reduce_individually(puzzle, minimal_clues, removed_clues, must_have)
        if len(minimal_clues) == minimal_clues_size:
            # cannot reduce anymore
            break

    return minimal_clues, removed_clues


def question_generation(rng: Random, col_name, table_data):
    values_by_cols = {}
    for row in table_data:
        for idx, value in enumerate(row):
            c = col_name[idx]
            if c not in values_by_cols:
                values_by_cols[c] = []
            values_by_cols[c].append(value)

    questions_data = []
    for row in table_data:
        for cid, col in enumerate(col_name):
            if cid == 0:
                continue
            question = f"What is {col} of the person who lives in House {row[0]}?"
            options = values_by_cols[col][:]
            rng.shuffle(options)
            truth = row[cid]
            assert truth in options
            questions_data.append(
                {"question": question, "choices": options, "truth_idx": options.index(truth), "answer": truth}
            )
            assert questions_data[-1]["answer"] in questions_data[-1]["choices"]
            assert questions_data[-1]["choices"][questions_data[-1]["truth_idx"]] == questions_data[-1]["answer"]

    return questions_data


def generate_solution_dict(rng: Random, selected_elements: list[Literal], n: int) -> OrderedDict[Literal, int]:
    solution = OrderedDict()
    house_ids = list(range(1, n + 1))
    for element in selected_elements:
        rng.shuffle(house_ids)
        attributes: list[Literal] = sorted(element.__members__.values())
        for i in range(n):
            solution[attributes[i]] = house_ids[i]
    return solution


def wrap_up_dict(rng: Random, random_elements, solution, puzzle, reduced, extra_clues, context, K, M):
    col_names = [e.__name__ for e in random_elements]
    house_data = OrderedDict()
    for item, house in solution.items():
        element_name, attrname = str(item).split(".")
        if house not in house_data:
            house_data[house] = OrderedDict()
        house_data[house][element_name] = attrname
    table_data = []
    for i in range(1, len(house_data) + 1):
        row = [i]
        for c in col_names:
            row.append(house_data[i][c].replace("_", " "))
        table_data.append(row)

    col_names = ["House"] + col_names

    table = tabulate(table_data, headers=col_names, tablefmt="grid")

    ## Generate multiple-choice questions
    q_data = question_generation(rng, col_names, table_data)
    all_in_one = OrderedDict()
    all_in_one["size"] = f"{K}*{M}"
    all_in_one["puzzle_context"] = context
    all_in_one["core_rules"] = [str(clue) for clue in reduced]
    all_in_one["extra_rules"] = [str(clue) for clue in extra_clues]
    all_in_one["core_rules_types"] = [str(type(clue)) for clue in reduced]
    all_in_one["extra_rules_types"] = [str(type(clue)) for clue in extra_clues]
    all_in_one["puzzle"] = str(puzzle)
    all_in_one["questions"] = q_data
    all_in_one["solution"] = OrderedDict(
        (("table_str", table), ("table_rows", table_data), ("table_header", col_names))
    )
    return all_in_one


def check_correctness(p: Puzzle) -> bool:
    solutions = itersolve(p.as_cnf())
    _first_solution = next(solutions)
    solution_set = [f"{str(k)} {v}" for k, v in p.solution.items()]
    return set(solution_set) == set(_first_solution)


def generate_puzzle(rng: Random, K=2, M=3) -> tuple[OrderedDict, Puzzle]:
    elements = [Color, Nationality, Animal, Drink, Cigar, Food, Flower, PhoneModel, Children, Smoothie]
    clue_types = [
        generate_found_at,
        generate_same_house,
        generate_consecutive_beside,
    ]

    rng.shuffle(elements)
    random_elements = [Name] + elements[: M - 1]
    solution = generate_solution_dict(rng, random_elements, K)

    # set up the puzzle with default constraints
    puzzle = Puzzle(rng=rng, element_types=random_elements, elements=solution.keys(), n_houses=K).set_constraints()
    puzzle.solution = solution
    context = str(puzzle)

    # generate all the clues
    clues: set[Clue] = set()

    for generate_function in clue_types:
        clues = clues.union(generate_function(puzzle, solution))

    reduced, _ = reduce_clues(puzzle, clues)
    extra_clues = clues - reduced
    extra_clues = set(rng.sample(list(extra_clues), min(len(extra_clues), 30)))
    for clue in reduced:
        puzzle.add_clue(clue)

    assert has_unique_solution(puzzle, puzzle.clues)
    assert check_correctness(puzzle)
    all_in_one = wrap_up_dict(rng, random_elements, solution, puzzle, reduced, extra_clues, context, K, M)
    return all_in_one, puzzle

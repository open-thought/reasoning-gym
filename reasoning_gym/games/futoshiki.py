"""Futoshiki puzzle generator"""

import copy
import itertools
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset


@dataclass
class FutoshikiConfig:
    """Configuration for Futoshiki puzzle generation"""

    board_size: int = 4  # Board will be NxN where N is this value
    difficulty: int = 1  # Possible values: 0, 1, 2, 3
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self):
        """Validate configuration parameters"""
        assert 4 <= self.board_size <= 9, "board_size must be between 4 and 9"
        assert 0 <= self.difficulty <= 3, "difficulty must be between 0 and 3"


class FutoshikiDataset(ProceduralDataset):
    """Generates Futoshiki puzzles with configurable board size and difficulty"""

    def __init__(self, config: FutoshikiConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __len__(self) -> int:
        return self.config.size

    def __iter__(self):
        self._current_idx = 0
        return self

    def __next__(self):
        if self._current_idx >= self.config.size:
            raise StopIteration
        item = self[self._current_idx]
        self._current_idx += 1
        return item

    def __getitem__(self, idx: int) -> dict:
        """
        Generate a single Futoshiki puzzle with blanks, represented by 0s, and constraints.
        Clues are pre-filled numbers in the grid.
        Constraints are adjacent cell pairs which may have '<' or '>' relations.
        Difficulty in [0..3] affects number of clues and constraints.
        """
        rng = Random(self.seed + idx)

        # Generate random "solved" Futoshiki grid
        solution = self._generate_random_solution(self.config.board_size, rng)
        # Add random adjacency constraints consistent with generated solved grid
        constraints = self._generate_random_constraints(solution, self.config.difficulty, rng)
        # Starting with full solution, remove clues to desired difficulty
        puzzle = self._remove_clues(copy.deepcopy(solution), constraints, rng)

        # Format as strings
        puzzle_str = self._puzzle_to_string(puzzle, constraints)
        solution_str = self._puzzle_to_string(solution, constraints)

        question = (
            f"Solve the following {self.config.board_size}x{self.config.board_size} Futoshiki puzzle:\n\n"
            f"{puzzle_str}\n\n"
            "Ensure your answer follows the same format as the puzzle above, just replace blanks (_) with the correct value for the cell.\n"
            "Use < and > for horizontal constraints. Use \u2227 and \u2228 for vertical constraints.\n"
            f"Remember, in Futoshiki each row and column must contain each number from 1 to {self.config.board_size} exactly once."
        )

        return {
            "question": question,
            "answer": solution_str,
            "metadata": {
                "puzzle": puzzle,
                "constraints": constraints,
                "solution": solution,
                "board_size": self.config.board_size,
                "difficulty": self.config.difficulty,
            },
        }

    def _puzzle_to_string(
        self, puzzle_grid: list[list[int]], constraints: dict[tuple[tuple[int, int], tuple[int, int]], str]
    ) -> str:
        """
        Formats a Futoshiki puzzle grid as a string with constraints.
        Constraints are represented as '<', '>', '\u2227', or '\u2228' between adjacent cells.
        """
        n = len(puzzle_grid)

        def cell_str(val: int) -> str:
            return str(val) if val != 0 else "_"

        # Helper to look up constraints between two adjacent cells
        # Ensures the first tuple is always the “lesser” in row-major order
        # If order is reversed in the dict, invert the constraint
        def get_constraint(r1, c1, r2, c2) -> Optional[str]:
            if (r1, c1) == (r2, c2):
                return None
            if (r1, c1) < (r2, c2):
                key = ((r1, c1), (r2, c2))
                sign = constraints.get(key)
                if sign == ">":  # first is bigger
                    if r1 == r2:  # horizontal
                        return ">"
                    else:  # vertical
                        return "\u2228"
                elif sign == "<":  # first is smaller
                    if r1 == r2:  # horizontal
                        return "<"
                    else:
                        return "\u2227"
            else:
                # reversed order in the dictionary -> invert the sign
                key = ((r2, c2), (r1, c1))
                sign = constraints.get(key)
                if sign == ">":
                    if r1 == r2:
                        return "<"
                    else:
                        return "\u2227"
                elif sign == "<":
                    if r1 == r2:
                        return ">"
                    else:
                        return "\u2228"
            return None

        lines = []

        for r in range(n):
            # Build the row string with horizontal constraints
            row_cells = []
            for c in range(n):
                row_cells.append(cell_str(puzzle_grid[r][c]))
                if c < n - 1:
                    hc = get_constraint(r, c, r, c + 1)
                    row_cells.append(hc if hc else " ")
            lines.append(" ".join(row_cells))

            # If not the last row, build the line of vertical constraints
            if r < n - 1:
                vert_cells = []
                for c in range(n):
                    vc = get_constraint(r, c, r + 1, c)
                    if vc:
                        vert_cells.append(vc)
                    else:
                        vert_cells.append(" ")
                    # Space out columns so vertical symbols line up under the correct spot
                    if c < n - 1:
                        vert_cells.append(" ")
                lines.append(" ".join(vert_cells))

        return "\n".join(lines)

    def _solve_logical(
        self,
        grid: list[list[int]],
        constraints: dict[tuple[tuple[int, int], tuple[int, int]], str],
    ) -> tuple[list[list[int]], list[list[set[int]]]]:
        """
        Apply logical rules to progress solution.
        Returns current state if logical rules can't progress further.
        Logical rules are implemented based on the descriptions here: https://futoshiki.uk/
        """
        size, working_grid = len(grid), copy.deepcopy(grid)

        # Starting point all numbers are candidates for all unfilled squares
        candidates: list[list[set[int]]] = [
            [set(range(1, len(grid) + 1)) if grid[r][c] == 0 else {grid[r][c]} for c in range(len(grid))]
            for r in range(len(grid))
        ]

        # Any cells > another cannot be 1, and any cells < another cannot be `size`
        # This is separated from the repeated function below to avoid redundant checks
        for ((r1, c1), (_, _)), rel in constraints.items():
            if rel == ">":
                candidates[r1][c1].discard(1)
            elif rel == "<":
                candidates[r1][c1].discard(size)

        def _update_grid():
            """Update solution wherever a cell's candidates set is reduced to a single element."""
            for r in range(len(working_grid)):
                for c in range(len(working_grid)):
                    if working_grid[r][c] == 0 and len(candidates[r][c]) == 1:
                        working_grid[r][c] = next(iter(candidates[r][c]))

        def _try_solve_logical() -> bool:
            progress = False

            # Eliminate candidates based on numbers already placed
            # If a number is placed in a cell, it cannot be a candidate for any other cell in the same row or column
            for r in range(len(working_grid)):
                for c in range(len(working_grid)):
                    if working_grid[r][c] == 0:
                        continue
                    for cc in range(len(working_grid)):
                        if cc != c and working_grid[r][c] in candidates[r][cc]:
                            candidates[r][cc].discard(working_grid[r][c])
                            progress = True
                    for rr in range(len(working_grid)):
                        if rr != r and working_grid[r][c] in candidates[rr][c]:
                            candidates[rr][c].discard(working_grid[r][c])
                            progress = True

            _update_grid()

            # Eliminate candidates based on constraints
            # Based on currently filled values, eliminate candidates that violate constraints
            def _eliminate_by_constraint(rc_less: tuple[int, int], rc_greater: tuple[int, int]) -> bool:
                r_less, c_less = rc_less
                r_greater, c_greater = rc_greater
                progress = False

                if working_grid[r_less][c_less] != 0:
                    # greater must only have candidates > less
                    for v in candidates[r_greater][c_greater].copy():
                        if v <= working_grid[r_less][c_less] and v in candidates[r_greater][c_greater]:
                            candidates[r_greater][c_greater].discard(v)
                            progress = True

                if working_grid[r_greater][c_greater] != 0:
                    # less must only have candidates < greater
                    for v in candidates[r_less][c_less].copy():
                        if v >= working_grid[r_greater][c_greater] and v in candidates[r_less][c_less]:
                            candidates[r_less][c_less].discard(v)
                            progress = True

                return progress

            for ((r1, c1), (r2, c2)), rel in constraints.items():
                v1, v2 = working_grid[r1][c1], working_grid[r2][c2]
                if v1 != 0 and v2 != 0:  # both already filled, skip
                    continue
                if rel == "<":
                    progress |= _eliminate_by_constraint((r1, c1), (r2, c2))
                elif rel == ">":
                    progress |= _eliminate_by_constraint((r2, c2), (r1, c1))

            _update_grid()

            # Seek "hidden singles" - cells where a candidate is unique in the row or column
            for r in range(len(working_grid)):
                for c in range(len(working_grid)):
                    if working_grid[r][c] != 0:
                        continue
                    for v in candidates[r][c]:
                        if sum(v in candidates[r][cc] for cc in range(len(working_grid))) == 1:
                            candidates[r][c] = {v}  # candidate unique in row
                            break
                        if sum(v in candidates[rr][c] for rr in range(len(working_grid))) == 1:
                            candidates[r][c] = {v}  # candidate unique in column
                            break

            _update_grid()

            # Seek "naked pairs" if same pair of candidates twice in a row or col, with nothing else in those two cells
            # Remove them from other cells in row/col
            for r in range(len(working_grid)):
                for c in range(len(working_grid)):
                    if working_grid[r][c] != 0 or len(candidates[r][c]) != 2:
                        continue
                    for cc in range(len(working_grid)):
                        if cc != c and candidates[r][cc] == candidates[r][c]:
                            for ccc in range(len(working_grid)):
                                if ccc != c and ccc != cc and candidates[r][c].intersection(candidates[r][ccc]):
                                    candidates[r][ccc] -= candidates[r][c]
                                    progress = True
                    for rr in range(len(working_grid)):
                        if rr != r and candidates[rr][c] == candidates[r][c]:
                            for rrr in range(len(working_grid)):
                                if rrr != r and rrr != rr and candidates[r][c].intersection(candidates[rrr][c]):
                                    candidates[rrr][c] -= candidates[r][c]
                                    progress = True

            _update_grid()

            # Seek "hidden pairs" - same pair of candidates occurs in two cells in a line, but nowhere else in the line
            # alongside other candidates (hence hidden). All other candidates can be removed from those two cells
            for r in range(len(working_grid)):
                for c in range(len(working_grid)):
                    if working_grid[r][c] != 0:
                        continue
                    for cc in range(c + 1, len(working_grid)):
                        if working_grid[r][cc] != 0:
                            continue
                        # Check if r, c shares a candidate pair with r, cc (maybe subset, not exact candidate set match)
                        r_c_pairs = itertools.permutations(candidates[r][c], 2)
                        r_cc_pairs = itertools.permutations(candidates[r][cc], 2)
                        for pair in r_c_pairs:
                            if pair not in r_cc_pairs:
                                continue
                            otherwise_unique = True
                            # If this pair occurs elsewhere in the row, skip
                            for ccc in range(len(working_grid)):
                                if ccc in (c, cc):
                                    continue
                                if pair in itertools.permutations(candidates[r][ccc], 2):
                                    otherwise_unique = False
                                    break
                            if not otherwise_unique:
                                continue
                            # Found a hidden pair, remove all other candidates from these two cells
                            candidates[r][c] = set(pair)
                            candidates[r][cc] = set(pair)

                    for rr in range(r + 1, len(working_grid)):
                        if working_grid[rr][c] != 0:
                            continue
                        # Check if r, c shares a candidate pair with rr, c (maybe subset, not exact candidate set match)
                        r_c_pairs = itertools.permutations(candidates[r][c], 2)
                        rr_c_pairs = itertools.permutations(candidates[rr][c], 2)
                        for pair in r_c_pairs:
                            if pair not in rr_c_pairs:
                                continue
                            otherwise_unique = True
                            # If this pair occurs elsewhere in the col, skip
                            for rrr in range(len(working_grid)):
                                if rrr in (r, rr):
                                    continue
                                if pair in itertools.permutations(candidates[rrr][c], 2):
                                    otherwise_unique = False
                                    break
                            if not otherwise_unique:
                                continue
                            # Found a hidden pair, remove all other candidates from these two cells
                            candidates[r][c] = set(pair)
                            candidates[rr][c] = set(pair)

            _update_grid()

            # Seek X-wings by rows
            for v in range(1, size + 1):
                # If candidate is in the same 2 positions in 2 rows, and nowhere else in those rows
                # Delete from the 2 intersecting cols

                # Find rows which have exactly 2 instances of the value in their candidate sets
                rows_with_v = [r for r in range(size) if sum(v in candidates[r][c] for c in range(size)) == 2]
                if len(rows_with_v) < 2:
                    continue
                # Check whether the 2 columns with the value are the same in the 2 rows
                cols_with_v_per_row = [set() for _ in range(len(rows_with_v))]
                for i, r in enumerate(rows_with_v):
                    for c in range(size):
                        if v in candidates[r][c]:
                            cols_with_v_per_row[i].add(c)
                # Check if there are a pair of tows with the same cols (there may be more than 2 rows)
                for i in range(len(rows_with_v)):
                    for j in range(i + 1, len(rows_with_v)):
                        if cols_with_v_per_row[i] == cols_with_v_per_row[j]:
                            # Found an X-wing, remove candidate from the 2 cols
                            for c in cols_with_v_per_row[i]:
                                for rr in range(size):
                                    if rr not in (rows_with_v[i], rows_with_v[j]) and v in candidates[rr][c]:
                                        candidates[rr][c].discard(v)
                                        progress = True

            # Seek X-wings by cols
            for v in range(1, size + 1):
                # If candidate is in the same 2 positions in 2 cols, and nowhere else in those cols
                # Delete from the 2 intersecting rows

                # Find cols which have exactly 2 instances of the value in their candidate sets
                cols_with_v = [c for c in range(size) if sum(v in candidates[r][c] for r in range(size)) == 2]
                if len(cols_with_v) < 2:
                    continue
                # Check whether the 2 rows with the value are the same in the 2 cols
                rows_with_v_per_col = [set() for _ in range(len(cols_with_v))]
                for i, c in enumerate(cols_with_v):
                    for r in range(size):
                        if v in candidates[r][c]:
                            rows_with_v_per_col[i].add(r)
                # Check if there are a pair of cols with the same rows (there may be more than 2 cols)
                for i in range(len(cols_with_v)):
                    for j in range(i + 1, len(cols_with_v)):
                        if rows_with_v_per_col[i] == rows_with_v_per_col[j]:
                            # Found an X-wing, remove candidate from the 2 rows
                            for r in rows_with_v_per_col[i]:
                                for cc in range(size):
                                    if cc not in (cols_with_v[i], cols_with_v[j]) and v in candidates[r][cc]:
                                        candidates[r][cc].discard(v)
                                        progress = True

            _update_grid()

            return progress

        while _try_solve_logical():
            continue

        return working_grid, candidates

    def _solve(
        self,
        grid: list[list[int]],
        constraints: dict[tuple[tuple[int, int], tuple[int, int]], str],
    ) -> list[list[int]] | None:
        """
        Backtracking Futoshiki solver. Used to verify generated puzzles.
        Applies logical rules first then backtracks to fill gaps.
        Return solved grid, or None if unsolvable.
        """

        grid, candidates = self._solve_logical(grid, constraints)

        size = len(grid)
        empty_cell = None

        # Find first empty cell
        for r in range(size):
            for c in range(size):
                if grid[r][c] == 0:
                    empty_cell = (r, c)
                    break
            if empty_cell:
                break

        # If no empty cell, solution is complete
        if not empty_cell:
            return copy.deepcopy(grid)

        r, c = empty_cell
        for val in range(1, size + 1):
            if val not in candidates[r][c]:
                continue
            if self._is_valid(grid, r, c, val, constraints):
                grid[r][c] = val
                solution = self._solve(grid, constraints)
                if solution is not None:
                    grid[r][c] = 0
                    return solution
                grid[r][c] = 0

        return None

    def _is_valid(
        self,
        grid: list[list[int]],
        row: int,
        col: int,
        val: int,
        constraints: dict[tuple[tuple[int, int], tuple[int, int]], str],
    ) -> bool:
        """Check row, col, and inequality constraints for placing val in grid[row][col]."""
        size = len(grid)

        # Row or column conflict?
        if val in grid[row]:
            return False
        for r in range(size):
            if grid[r][col] == val:
                return False

        # Temporarily place the val and check constraints
        original_val = grid[row][col]
        grid[row][col] = val

        # Check all constraints involving this cell
        for ((r1, c1), (r2, c2)), rel in constraints.items():
            v1 = grid[r1][c1]
            v2 = grid[r2][c2]
            # If either is 0, skip
            if v1 == 0 or v2 == 0:
                continue
            # If relation is '<', v1 < v2 must hold
            if rel == "<":
                if not (v1 < v2):
                    grid[row][col] = original_val
                    return False
            elif rel == ">":
                if not (v1 > v2):
                    grid[row][col] = original_val
                    return False

        grid[row][col] = original_val
        return True

    def _generate_random_solution(self, size: int, rng: Random) -> list[list[int]]:
        """
        Generates a random valid completed Futoshiki solution with numbers 1..size.
        Ensures each row and column has unique numbers.
        """
        # Fill row by row with a random permutation, ensuring no column conflicts. Use backtracking
        grid = [[0] * size for _ in range(size)]

        def backtrack(r):
            if r == size:
                return True
            nums = list(range(1, size + 1))
            rng.shuffle(nums)
            for permutation in itertools.permutations(nums):
                # Place row if columns are valid
                valid = True
                for c in range(size):
                    if any(grid[rr][c] == permutation[c] for rr in range(r)):
                        valid = False
                        break
                if valid:
                    grid[r] = list(permutation)
                    if backtrack(r + 1):
                        return True
            return False

        if backtrack(0):
            return grid

        raise ValueError("Could not generate a random solution.")

    def _generate_random_constraints(
        self, solution: list[list[int]], difficulty: int, rng: Random
    ) -> dict[tuple[tuple[int, int], tuple[int, int]], str]:
        """
        Randomly add inequality constraints that match the solution.
        We only add constraints for adjacent cells (horizontal or vertical).
        Probability of adding a constraint can scale with difficulty.
        """
        size = len(solution)
        constraints = {}
        # For each pair of adjacent cells, we might add a constraint
        # P(adding a constraint) increases with difficulty on an arbitrary scale
        base_prob = 0.03 + 0.07 * difficulty
        for r in range(size):
            for c in range(size):
                # Horizontal neighbor
                if c < size - 1:
                    if rng.random() < base_prob:
                        if solution[r][c] < solution[r][c + 1]:
                            constraints[((r, c), (r, c + 1))] = "<"
                        else:
                            constraints[((r, c), (r, c + 1))] = ">"
                # Vertical neighbor
                if r < size - 1:
                    if rng.random() < base_prob:
                        if solution[r][c] < solution[r + 1][c]:
                            constraints[((r, c), (r + 1, c))] = "<"
                        else:
                            constraints[((r, c), (r + 1, c))] = ">"
        return constraints

    def count_solutions(self, grid, constraints, limit=2) -> int:
        size = len(grid)
        count = 0

        def backtrack():
            nonlocal count
            # Early exit if limit reached
            if count >= limit:
                return
            # Find the next empty cell
            for r in range(size):
                for c in range(size):
                    if grid[r][c] == 0:
                        for val in range(1, size + 1):
                            if self._is_valid(grid, r, c, val, constraints):
                                grid[r][c] = val
                                backtrack()
                                grid[r][c] = 0
                        return
            count += 1

        backtrack()
        return count

    def _remove_clues(
        self,
        grid: list[list[int]],
        constraints: dict[tuple[tuple[int, int], tuple[int, int]], str],
        rng: Random,
    ) -> list[list[int]]:
        """
        Remove clues from a full solution to try to maintain a unique-solution puzzle.
        We remove in random order until we reach our target, or can't without losing uniqueness.
        """
        size = len(grid)
        fill_fraction = 0.1
        target_filled = int(fill_fraction * (size * size))

        coords = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(coords)

        def _count_filled_cells(g):
            return sum(g[r][c] != 0 for r in range(size) for c in range(size))

        def _try_remove():
            for r, c in coords:
                if _count_filled_cells(grid) <= target_filled:
                    break  # Removal target hit

                saved = grid[r][c]
                if saved == 0:
                    continue
                # Try remove
                grid[r][c] = 0

                # Check if unsolvable
                sol = self._solve(copy.deepcopy(grid), constraints)
                if sol is None:
                    grid[r][c] = saved
                    continue
                # Check if not unique
                if self.count_solutions(copy.deepcopy(grid), constraints, limit=2) > 1:
                    grid[r][c] = saved

        _try_remove()

        # Second pass if we aren't close to target
        if _count_filled_cells(grid) > 2 * target_filled:
            rng.shuffle(coords)
            _try_remove()

        return grid

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        if not answer:
            return 0.0

        oracle_answer = entry["answer"]
        metadata = entry["metadata"]
        solution: list[list[int]] = metadata["solution"]
        board_size: int = len(solution[0])

        # 1. match answer without trailing whitespaces
        answer_stripped = "\n".join(l.rstrip() for l in answer.split("\n"))
        oracle_answer_stripped = "\n".join(l.rstrip() for l in oracle_answer.split("\n"))

        if answer_stripped == oracle_answer_stripped:
            reward = 1.0
        else:
            # 2. accept answers with correct numeric sequence (ignoring non-numeric characters)
            row = 0
            num_matching = 0
            for ln in answer.split("\n"):
                if row >= len(solution):
                    break
                numbers = [int(c) for c in ln if c in "123456789"]
                if len(numbers) != len(solution[0]):
                    continue  # ignore lines without numbers
                for a, b in zip(solution[row], numbers):
                    if a == b:
                        num_matching += 1
                row += 1

            reward = num_matching / (board_size * board_size)
            reward *= 0.9  # penalty for not using standard format

        if len(answer) > len(oracle_answer):
            reward *= len(oracle_answer) / len(answer)  # penalty for additional length
        return reward


register_dataset("futoshiki", FutoshikiDataset, FutoshikiConfig)

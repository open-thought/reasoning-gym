"""Futoshiki puzzle generator"""

import copy
from dataclasses import dataclass
from random import Random
from typing import Dict, List, Optional, Set, Tuple

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

        puzzle, constraints, solution = self._generate_futoshiki(
            self.config.board_size, (0.02 + (0.02 * self.config.difficulty)), rng
        )

        # Format as strings
        puzzle_str = self._puzzle_to_string(puzzle, constraints)
        solution_str = self._puzzle_to_string(solution, constraints)

        return {
            "question": f"Solve the following Futoshiki puzzle:\n{puzzle_str}",
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
        self,
        puzzle_grid: List[List[int]], 
        constraints: List[List[str]],
    ) -> str:
        constraint_matrix = self._build_constraint_matrix(puzzle_grid, constraints)

        n = len(puzzle_grid)

        def cell_str(val: int) -> str:
            return str(val) if val != 0 else "_"

        def display_constraint(sign: str, vertical: bool, inverted: bool) -> str:
            if not vertical:
                if not inverted:
                    return sign
                else:
                    return "<" if sign == ">" else ">" if sign == "<" else sign
            else:
                if not inverted:
                    return "\u2227" if sign == ">" else "\u2228" if sign == "<" else sign
                else:
                    return "\u2228" if sign == ">" else "\u2227" if sign == "<" else sign

        def get_constraint(r1: int, c1: int, r2: int, c2: int) -> Optional[str]:
            if (r1, c1) == (r2, c2):
                return None

            # Horizontal neighbors
            if r1 == r2:
                if c2 == c1 + 1:
                    # (r1, c1) is the lesser cell. Use its right constraint directly
                    sign = constraint_matrix[r1][c1][0]
                    return display_constraint(sign, vertical=False, inverted=False) if sign else None
                elif c2 == c1 - 1:
                    # (r2, c2) is the lesser cell. Use its right constraint and invert
                    sign = constraint_matrix[r2][c2][0]
                    return display_constraint(sign, vertical=False, inverted=True) if sign else None

            # Vertical neighbors
            elif c1 == c2:
                if r2 == r1 + 1:
                    # (r1, c1) is the lesser cell. Use its below constraint directly
                    sign = constraint_matrix[r1][c1][1]
                    return display_constraint(sign, vertical=True, inverted=False) if sign else None
                elif r2 == r1 - 1:
                    # (r2, c2) is the lesser cell. Use its below constraint and invert
                    sign = constraint_matrix[r2][c2][1]
                    return display_constraint(sign, vertical=True, inverted=True) if sign else None

            return None

        lines = []

        for r in range(n):
            row_cells = []
            for c in range(n):
                row_cells.append(cell_str(puzzle_grid[r][c]))
                if c < n - 1:
                    # Get the horizontal constraint between (r, c) and (r, c+1)
                    hc = get_constraint(r, c, r, c + 1)
                    row_cells.append(hc if hc is not None else " ")
            lines.append(" ".join(row_cells))

            if r < n - 1:
                vert_cells = []
                for c in range(n):
                    # Get the vertical constraint between (r, c) and (r+1, c)
                    vc = get_constraint(r, c, r + 1, c)
                    vert_cells.append(vc if vc is not None else " ")
                    if c < n - 1:
                        vert_cells.append(" ")
                lines.append(" ".join(vert_cells))

        return "\n".join(lines)

    def _build_cell_lookup(self, candidates: List[List[Set]]) -> Dict[int, List[Tuple[int, int]]]:
        size = len(candidates)
        cell_lookup = {m: [] for m in range(1, size + 1)}

        for j in range(size):
            for i in range(size):
                if len(candidates[j][i]) > 0:
                    cell_lookup[len(candidates[j][i])].append((j, i))

        return cell_lookup


    def _build_constraint_matrix(self, grid: List[List[int]], constraints: List[List[str]]) -> List[List[List[str]]]:
        """Build a matrix of constraints from the original human-readable constraints."""
        size = len(grid)
        constraint_matrix = [[[None] * 4 for _ in range(size)] for _ in range(size)]

        def _try_fill(j, i, direction, x, y):
            try:
                constraint_matrix[j][i][direction] = constraints[x][y]
            except IndexError:
                constraint_matrix[j][i][direction] = None

        for j in range(size):
            for i in range(size):
                _try_fill(j, i, 0, j * 2, i)  # right
                _try_fill(j, i, 1, (j * 2) + 1, i)  # below
                _try_fill(j, i, 2, j * 2, i - 1)  # left
                _try_fill(j, i, 3, (j * 2) - 1, i)  # above

        return constraint_matrix


    def _update_candidates(self, grid: List[List[int]], candidates: List[List[Set]]):
        """Use current filled values to reduce candidates in lines/columns."""
        size = len(grid)
        for j in range(size):
            for i in range(size):
                if grid[j][i] == 0:
                    continue

                candidates[j][i] = set([grid[j][i]])

                for k in range(size):
                    if k != i:
                        candidates[j][k].discard(grid[j][i])
                    if k != j:
                        candidates[k][i].discard(grid[j][i])


    def _update_grid(self, grid: List[List[int]], candidates: List[List[Set]]):
        """Update grid by inputting locations with a single possible value. Then update candidates."""
        size = len(grid)
        for j in range(size):
            for i in range(size):
                if grid[j][i] == 0 and len(candidates[j][i]) == 1:
                    grid[j][i] = next(iter(candidates[j][i]))
        self._update_candidates(grid, candidates)


    def _recursive_greater_than(self, j: int, i: int, grid: List[List[int]], candidates: List[List[Set]], constraint_matrix: List[List[List[str]]]):
        """Recursively follow greater than signs to filter down candidates."""

        def _try_update(j, i, a, b):
            try:
                candidates[j][i] = set([x for x in candidates[j][i] if x > min(candidates[a][b])])
            except ValueError:
                pass

        if constraint_matrix[j][i][0] == ">":
            self._recursive_greater_than(j, i + 1, grid, candidates, constraint_matrix)
            _try_update(j, i, j, i + 1)
            self._update_grid(grid, candidates)

        if constraint_matrix[j][i][1] == u"\u2228":
            self._recursive_greater_than(j + 1, i, grid, candidates, constraint_matrix)
            _try_update(j, i, j + 1, i)
            self._update_grid(grid, candidates)

        if constraint_matrix[j][i][2] == "<":
            self._recursive_greater_than(j, i - 1, grid, candidates, constraint_matrix)
            _try_update(j, i, j, i - 1)
            self._update_grid(grid, candidates)

        if constraint_matrix[j][i][3] == u"\u2227":
            self._recursive_greater_than(j - 1, i, grid, candidates, constraint_matrix)
            _try_update(j, i, j - 1, i)
            self._update_grid(grid, candidates)


    def _recursive_less_than(self, j: int, i: int, grid: List[List[int]], candidates: List[List[Set]], constraint_matrix: List[List[List[str]]]):
        """Recursively follow less than signs to filter down candidates."""

        def _try_update(j, i, a, b):
            try:
                candidates[j][i] = set([x for x in candidates[j][i] if x < max(candidates[a][b])])
            except ValueError:
                pass

        if constraint_matrix[j][i][0] == "<":
            self._recursive_less_than(j, i + 1, grid, candidates, constraint_matrix)
            _try_update(j, i, j, i + 1)
            self._update_grid(grid, candidates)

        if constraint_matrix[j][i][1] == u"\u2227":
            self._recursive_less_than(j + 1, i, grid, candidates, constraint_matrix)
            _try_update(j, i, j + 1, i)
            self._update_grid(grid, candidates)

        if constraint_matrix[j][i][2] == ">":
            self._recursive_less_than(j, i - 1, grid, candidates, constraint_matrix)
            _try_update(j, i, j, i - 1)
            self._update_grid(grid, candidates)

        if constraint_matrix[j][i][3] == u"\u2228":
            self._recursive_less_than(j - 1, i, grid, candidates, constraint_matrix)
            _try_update(j, i, j - 1, i)
            self._update_grid(grid, candidates)


    def _line_match(self, grid: List[List[int]], candidates: List[List[Set]], line: List[Set], match_seq: Set):
        """Find cells in a line (row/col) where candidates are the same. If matches are found, remove relevant candidates from all other cells."""
        size = len(line)
        line_matches = [idx for idx, val in enumerate(line) if val == match_seq]

        def _remove_matched(line_matches):
            unmatched_line_indices = [x for x in range(size) if x not in line_matches]
            for index in unmatched_line_indices:
                try:
                    for num in match_seq:
                        line[index].discard(num)
                except ValueError:
                    continue
            self._update_grid(grid, candidates)

        # Remove exact matches
        if len(line_matches) == len(match_seq):
            _remove_matched(line_matches)

        if len(match_seq) != 2 or len(line_matches) != 1:
            return

        # Look for partial matches
        third_values_attempted = []
        for item in copy.deepcopy(line):
            if item == match_seq:
                continue

            if len(item) == 3 and all(x in item for x in match_seq):
                # All of the match list is contained in new item
                third_val = [x for x in copy.deepcopy(item) if x not in match_seq]
                if third_val[0] in third_values_attempted:
                    continue

                third_values_attempted.append(third_val[0])
                new_match_value = list(copy.deepcopy(match_seq))
                new_match_value.append(third_val[0])
                new_match_value.sort()

                new_line_matches = [index for index, val in enumerate(line) if (val == new_match_value) or (val == match_seq)]

                if len(new_line_matches) == 3:
                    _remove_matched(new_line_matches)


    def _corner_rule(self, grid: List[List[int]], candidates: List[List[Set]], constraint_matrix: List[List[List[str]]], j: int, i: int):
        match_list = candidates[j][i]
        if len(match_list) != 2:
            return

        row = candidates[j].copy()
        row_matches = [index for index, val in enumerate(row) if val == match_list]
        if len(row_matches) != 2:
            return

        i1, i2 = row_matches[0], row_matches[1]
        if ((constraint_matrix[j][i1][1] == u"\u2227" and constraint_matrix[j][i2][3] == u"\u2228") and candidates[j+1][i1] == candidates[j-1][i2]):
            num_to_remove = max(candidates[j+1][i1])
            candidates[j+1][i2].discard(num_to_remove)
            candidates[j-1][i1].discard(num_to_remove)
            self._update_grid(grid, candidates)

        if (constraint_matrix[j][i1][3] == u"\u2228" and constraint_matrix[j][i2][1] == u"\u2227" and candidates[j-1][i1] == candidates[j+1][i2]):
            num_to_remove = max(candidates[j-1][i1])
            candidates[j+1][i1].discard(num_to_remove)
            candidates[j-1][i2].discard(num_to_remove)
            self._update_grid(grid, candidates)


    def _check_cell_validity(self, grid: List[List[int]], constraint_matrix: List[List[List[str]]], j: int, i: int) -> bool:
        """Test the current cell value against constraints."""
        neighbours = [
            grid[j][i + 1] if i + 1 < len(grid[j]) else None,
            grid[j + 1][i] if j + 1 < len(grid) else None,
            grid[j][i - 1] if i - 1 >= 0 else None,
            grid[j - 1][i] if j - 1 >= 0 else None,
        ]

        valid = True

        if constraint_matrix[j][i][0] == ">" and neighbours[0] is not None:
            valid = valid & (grid[j][i] > neighbours[0])
        if constraint_matrix[j][i][1] == u"\u2228" and neighbours[1] is not None:
            valid = valid & (grid[j][i] > neighbours[1])
        if constraint_matrix[j][i][2] == "<" and neighbours[2] is not None:
            valid = valid & (grid[j][i] > neighbours[2])
        if constraint_matrix[j][i][3] == u"\u2227" and neighbours[3] is not None:
            valid = valid & (grid[j][i] > neighbours[3])

        if constraint_matrix[j][i][0] == "<" and neighbours[0] is not None and neighbours[0] != 0:
            valid = valid & (grid[j][i] < neighbours[0])
        if constraint_matrix[j][i][1] == u"\u2227" and neighbours[1] is not None and neighbours[1] != 0:
            valid = valid & (grid[j][i] < neighbours[1])
        if constraint_matrix[j][i][2] == ">" and neighbours[2] is not None and neighbours[2] != 0:
            valid = valid & (grid[j][i] < neighbours[2])
        if constraint_matrix[j][i][3] == u"\u2228" and neighbours[3] is not None and neighbours[3] != 0:
            valid = valid & (grid[j][i] < neighbours[3])

        return valid


    def _check_validity(self, grid: List[List[int]], constraint_matrix: List[List[List[str]]]) -> bool:
        """Checks if the current grid is valid."""
        for j, line in enumerate(grid):
            for i, n in enumerate(line):
                if n == 0:
                    return False
                if not self._check_cell_validity(grid, constraint_matrix, j, i):
                    return False
            if not all([x in line for x in range(1, len(line) + 1)]):
                return False
        return True


    def _solve_logical(self, grid: List[List[int]], constraints: List[List[str]], max_iterations: int = 20) -> Tuple[List[List[int]], List[List[Set]], bool]:
        """Apply logical algorithms to solve the given puzzle."""

        iteration, size, solved = 0, len(grid), False
        working_grid = copy.deepcopy(grid)

        candidates = [[set(range(1, size + 1)) for _ in range(size)] for _ in range(size)]
        cell_lookup = self._build_cell_lookup(candidates)

        constraint_matrix = self._build_constraint_matrix(grid, constraints)

        while iteration < max_iterations and len(cell_lookup[1]) < size * size:
            for j in range(size):
                for i in range(size):
                    # Recursively follow constraints from this cell to filter candidates
                    self._recursive_greater_than(j, i, working_grid, candidates, constraint_matrix)
                    self._recursive_less_than(j, i, working_grid, candidates, constraint_matrix)

                    # Check if cell is the only possible location for any of its remaining candidate values
                    for candidate_val in candidates[j][i]:
                        unique_in_row, unique_in_col = True,  True
                        for k in range(size):
                            if k != i and candidate_val in candidates[j][k]:
                                unique_in_row = False
                            if k != j and candidate_val in candidates[k][i]:
                                unique_in_col = False

                        if unique_in_row or unique_in_col:
                            # This is the only possible cell for this candidate
                            candidates[j][i] = {candidate_val}
                            self._update_grid(working_grid, candidates)
                            break

                    # Find line matches for this row and column
                    row = copy.deepcopy(candidates[j])
                    self._line_match(grid, candidates, row, candidates[j][i])
                    col = copy.deepcopy([x[i] for x in candidates])
                    self._line_match(grid, candidates, col, candidates[j][i])

                    # Apply corner rule
                    self._corner_rule(grid, candidates, constraint_matrix, j, i)

            cell_lookup = self._build_cell_lookup(candidates)
            iteration += 1

        solved = self._check_validity(working_grid, constraint_matrix)

        if len(cell_lookup[1]) == size * size and solved:
            return working_grid, candidates, True

        return working_grid, candidates, False


    def _solve_brute_force(self, grid: List[List[int]], candidates: List[List[Set]], constraints: List[List[str]], rng: Random, max_depth: int = 9) -> List[List[int]] | None:
        """Try each possible value for each cell until a solution is found."""
        size = len(grid)

        cell_lookup = self._build_cell_lookup(candidates)

        for m in range(2, size):
            for cell in cell_lookup[m]:
                cell_candidates = list(candidates[cell[0]][cell[1]])
                for index in range(len(cell_candidates)):
                    test_puzzle = copy.deepcopy(grid)
                    test_puzzle[cell[0]][cell[1]] = cell_candidates[index]

                    # Use few iterations on logical solver when we have entered brute force
                    solution, candidates, solved = self._solve_logical(test_puzzle, constraints, max_iterations=3)

                    if solved:
                        return solution

                    if max_depth > 0:
                        return self._solve_brute_force(solution, candidates, constraints, rng, max_depth - 1)

        return None


    def _generate_futoshiki(self, size: int, constraint_factor: float, rng: Random) -> Tuple[List[List[int]], List[List[str]], List[List[int]]]:
        grid = [[0 for _ in range(size)] for _ in range(size)]
        constraints = []

        for j in range(2 * size - 1):
            line = []
            if (j + 1) % 2 == 0:  # Odd line
                for _ in range(size):
                    a = rng.random()
                    if a >= constraint_factor and a <= 1 - constraint_factor:
                        line.append("")
                    elif a < constraint_factor:
                        line.append(u"\u2228")
                    elif a > 1 - constraint_factor:
                        line.append(u"\u2227")
            else:  # Even line
                for _ in range(size - 1):
                    a = rng.random()
                    if a >= constraint_factor and a <= 1 - constraint_factor:
                        line.append("")
                    elif a < constraint_factor:
                        line.append("<")
                    elif a > 1 - constraint_factor:
                        line.append(">")
            constraints.append(line)

        try:
            solution, candidates, solved = self._solve_logical(grid, constraints, max_iterations=10)
        except KeyError:
            # Unsolvable puzzle, retry
            return self._generate_futoshiki(size, constraint_factor, rng)

        if not solved:
            solution = self._solve_brute_force(solution, candidates, constraints, rng)

            if solution is None:
                # Unsolvable puzzle, retry
                return self._generate_futoshiki(size, constraint_factor, rng)

        return grid, constraints, solution


register_dataset("futoshiki", FutoshikiDataset, FutoshikiConfig)

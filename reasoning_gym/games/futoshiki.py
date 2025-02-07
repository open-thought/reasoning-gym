"""Futoshiki puzzle generator"""

import copy
import itertools
import random
from dataclasses import dataclass
from random import Random
from typing import Dict, List, Optional, Tuple

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
        puzzle = self._remove_clues(copy.deepcopy(solution), constraints, self.config.difficulty, rng)

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
        constraints: Dict[Tuple[Tuple[int, int], Tuple[int, int]], str]
    ) -> str:
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
                if sign == ">":      # first is bigger
                    if r1 == r2:     # horizontal
                        return ">"
                    else:            # vertical
                        return "\u2227"
                elif sign == "<":    # first is smaller
                    if r1 == r2:     # horizontal
                        return "<"
                    else:
                        return "\u2228"
            else:
                # reversed order in the dictionary -> invert the sign
                key = ((r2, c2), (r1, c1))
                sign = constraints.get(key)
                if sign == ">":
                    if r1 == r2: 
                        return "<"
                    else:
                        return "\u2228"
                elif sign == "<":
                    if r1 == r2:
                        return ">"
                    else:
                        return "\u2227"
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

    # currently this gets a bit slow for larger grid sizes as it relies on brute force backtracking
    # possible improvements: implement optimisations, using common rules in Futoshiki to reduce search space
    # see: https://www.futoshiki.com/how-to-solve
    # also see other solvers' approaches e.g. https://github.com/davidswarbrick/futoshiki-solver/blob/master/Futoshiki.py
    # however I attempted some optimisations based on the code of the above parser, such as the recursive constraint following, and it was actually quite a lot slower

    def _solve(
        self,
        grid: List[List[int]],
        constraints: Dict[Tuple[Tuple[int, int], Tuple[int, int]], str],
        rng: Random,
        find_multiple: bool = False,
    ) -> List[List[int]] | None:
        """
        Backtracking Futoshiki solver. Used to verify generated puzzles.
        Return solved grid, or None if unsolvable.
        If find_multiple: also return None if more than one solution found.
        """
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
            if self._is_valid(grid, r, c, val, constraints):
                grid[r][c] = val
                solution = self._solve(grid, constraints, rng, find_multiple)
                if solution is not None:
                    # If find_multiple, continue searching to check for non-uniqueness
                    if find_multiple and self._has_other_solution(solution, grid, constraints, rng):
                        grid[r][c] = 0
                        return None

                    grid[r][c] = 0
                    return solution
                grid[r][c] = 0
        
        return None

    def _has_other_solution(
        self,
        existing_solution: List[List[int]],
        partial_grid: List[List[int]],
        constraints: Dict[Tuple[Tuple[int, int], Tuple[int, int]], str],
        rng: Random,
    ) -> bool:
        """
        Check if there's at least one solution different from existing_solution, given the partial_grid so far.
        This is a quick hack: we attempt to find another solution with a slight difference.
        A full approach is backtracking that tries to find any solution differing from existing_solution in >= 1 cell.
        """
        # Each cell not set in partial_grid could be varied
        size = len(existing_solution)
        # Make a fresh puzzle using partial_grid
        puzzle_copy = copy.deepcopy(partial_grid)
        
        def backtrack(i = 0, j = 0) -> bool:
            # Move past end of row
            if j == size:
                i += 1
                j = 0
            # Completed all rows
            if i == size:
                # Confirm puzzle_copy differs in at least one cell from existing_solution
                for rr in range(size):
                    for cc in range(size):
                        if puzzle_copy[rr][cc] != existing_solution[rr][cc]:
                            return True
                return False
            
            if puzzle_copy[i][j] != 0:
                # Move on
                return backtrack(i, j + 1)
            else:
                # Try different values
                vals = list(range(1, size + 1))
                rng.shuffle(vals)
                for val in vals:
                    if self._is_valid(puzzle_copy, i, j, val, constraints):
                        puzzle_copy[i][j] = val
                        if backtrack(i, j + 1):
                            return True
                        puzzle_copy[i][j] = 0
                return False

        return backtrack(0, 0)

    def _is_valid(
        self,
        grid: List[List[int]],
        row: int,
        col: int,
        val: int,
        constraints: Dict[Tuple[Tuple[int, int], Tuple[int, int]], str]
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

    def _generate_random_solution(self, size: int, rng: Random) -> List[List[int]]:
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
        self, solution: List[List[int]], difficulty: int, rng: Random
    ) -> Dict[Tuple[Tuple[int, int], Tuple[int,int]], str]:
        """
        Randomly add inequality constraints that match the solution.
        We only add constraints for adjacent cells (horizontal or vertical).
        Probability of adding a constraint can scale with difficulty.
        """
        size = len(solution)
        constraints = {}
        # For each pair of adjacent cells, we might add a constraint
        # P(adding a constraint) increases with difficulty on an arbitrary scale
        base_prob = 0.05 + 0.05 * difficulty
        for r in range(size):
            for c in range(size):
                # Horizontal neighbor
                if c < size - 1:
                    if rng.random() < base_prob:
                        if solution[r][c] < solution[r][c+1]:
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

    def _remove_clues(
        self,
        grid: List[List[int]],
        constraints: Dict[Tuple[Tuple[int, int], Tuple[int, int]], str],
        difficulty: int,
        rng: Random,
    ) -> List[List[int]]:
        """
        Remove clues from a full solution to try to maintain a unique-solution puzzle.
        The higher the difficulty, the more clues we remove.
        We remove in random order until we reach our target, or can't without losing uniqueness.
        """
        size = len(grid)
        fill_fraction = [0.09, 0.07, 0.05, 0.03]  # Easiest -> hardest
        target_filled = int(fill_fraction[difficulty] * (size * size))

        coords = [(r, c) for r in range(size) for c in range(size)]
        rng.shuffle(coords)

        def count_filled_cells(g):
            return sum(g[r][c] != 0 for r in range(size) for c in range(size))

        for (r,c) in coords:
            if count_filled_cells(grid) <= target_filled:
                break  # Removal target hit

            saved = grid[r][c]
            if saved == 0:
                continue
            # Try remove
            grid[r][c] = 0

            # Check if unsolvable or non-unique
            puzzle_copy = copy.deepcopy(grid)
            sol = self._solve(puzzle_copy, constraints, rng, find_multiple=True)
            if sol is None:
                # Not solvable or non-unique, revert
                grid[r][c] = saved

        return grid


register_dataset("futoshiki", FutoshikiDataset, FutoshikiConfig)

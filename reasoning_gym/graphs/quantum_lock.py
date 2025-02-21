import re
from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset


@dataclass
class QuantumLockConfig:
    """Configuration for QuantumLock task generation"""

    difficulty: int = 10
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.difficulty > 0, "difficulty must be positive"
        assert self.size > 0, "size must be positive"


class QuantumLockDataset(ProceduralDataset):
    """Generates QuantumLock tasks"""

    def __init__(self, config: QuantumLockConfig):
        self._prompt_templates = [
            """\
In front of you are some buttons, a light, and a number. The light will toggle between red and green whenever you press a button. Each button performs a mathematical operation to the number, but the operation may depend on the state of the light.
You must press the shortest correct sequence of buttons to reach the target value. Your answer should be a sequence of buttons separated by '→', for example: A → B → C

Start: {initial_value} ({initial_state})
Target: {target_value}
Buttons:
{buttons}"""
        ]
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single QuantumLock task

        Returns:
            dict with keys:
                - question: str, the task description
                - answer: str, a solution string
                - metadata: dict with generation parameters
        """
        rng = Random(self.seed + idx)

        puzzle_data = self.generate_quantum_puzzle(rng, self.config.difficulty)

        return {
            "question": self.format_puzzle(rng.choice(self._prompt_templates), puzzle=puzzle_data),
            "answer": " → ".join(puzzle_data["solution"]),
            "metadata": {
                "difficulty": self.config.difficulty,
                "solution_path": puzzle_data["solution"],
                "target_value": puzzle_data["target_value"],
                "buttons": puzzle_data["buttons"],
                "initial_state": puzzle_data["initial_state"],
                "initial_value": puzzle_data["initial_value"],
            },
        }

    def generate_quantum_puzzle(self, rng: Random, difficulty: int = 1) -> dict[str, Any]:
        """
        Generates a Quantum Lock puzzle with configurable difficulty.
        Returns a dictionary containing puzzle parameters and solution.
        """
        # Define operation parameters based on difficulty
        base_values = {
            "add": [2, 3] if difficulty >= 5 else [1, 2],
            "subtract": [2, 3] if difficulty >= 5 else [1, 2],
            "multiply": [2, 3] if difficulty >= 7 else [2],
        }

        operations = [
            {"type": "add", "values": base_values["add"]},
            {"type": "subtract", "values": base_values["subtract"]},
            {"type": "multiply", "values": base_values["multiply"]},
        ]

        # Generate unique buttons with collision protection
        buttons = []
        used_combinations = set()

        while len(buttons) < 3:
            op = rng.choice(operations)
            btn_value = rng.choice(op["values"])

            # State selection with weighted probabilities
            state_weights = {"any": 4, "green": 2, "red": 1}
            active_state = rng.choices(list(state_weights.keys()), weights=state_weights.values(), k=1)[0]

            # Create unique combination check
            combo = (op["type"], btn_value, active_state)
            if combo in used_combinations:
                continue

            # Prevent duplicate button effects
            if any(
                b["type"] == op["type"] and b["value"] == btn_value and b["active_state"] == active_state
                for b in buttons
            ):
                continue

            buttons.append(
                {"name": chr(65 + len(buttons)), "type": op["type"], "value": btn_value, "active_state": active_state}
            )
            used_combinations.add(combo)

        # Dynamic target scaling with non-linear progression
        base_target = 5 + (difficulty**1.5)
        variance = rng.randint(-int(base_target * 0.2), int(base_target * 0.3))
        target = max(8, int(base_target + variance))

        # Create puzzle structure
        puzzle = {
            "initial_value": 0,
            "initial_state": "red",
            "target_value": target,
            "buttons": buttons,
            "max_steps": min(15, 6 + int(difficulty * 1.5)),
            "solution": None,
        }

        # Find shortest solution using BFS
        queue = deque([(0, "red", [])])
        visited = set()

        while queue:
            val, state, path = queue.popleft()

            if val == puzzle["target_value"]:
                puzzle["solution"] = path
                return puzzle

            if len(path) >= puzzle["max_steps"] or (val, state) in visited:
                continue

            visited.add((val, state))

            for btn in buttons:
                next_state = "green" if state == "red" else "red"

                # Check if button is usable
                if btn["active_state"] not in [state, "any"]:
                    continue

                # Calculate new value
                try:
                    if btn["type"] == "add":
                        new_val = val + btn["value"]
                    elif btn["type"] == "subtract":
                        new_val = val - btn["value"]
                    elif btn["type"] == "multiply":
                        new_val = val * btn["value"]
                except:
                    continue  # Handle overflows if needed

                queue.append((new_val, next_state, path + [btn["name"]]))

        # If no solution found, regenerate
        return self.generate_quantum_puzzle(rng, difficulty)

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the task.

        The function awards 1.0 for a correct answer and less otherwise.
        """
        if answer == None:
            return 0.0

        # Get correct solution from metadata
        correct_solution = entry["metadata"].get("solution_path", [])

        # Normalize both answers
        def normalize_seq(seq):
            """Handle both string and list inputs by converting to string first"""
            # Convert sequence to string representation if it's a list
            input_str = "".join(seq) if isinstance(seq, list) else str(seq or "")
            return [c.upper() for c in re.findall(r"[A-C]", input_str.upper())]

        user_sequence = normalize_seq(answer)
        target_sequence = normalize_seq("".join(correct_solution))

        # Exact sequence match required
        if user_sequence == target_sequence:
            return 1.0

        # Partial credit for reaching target (optional)
        final_state = self.simulate_sequence(entry["metadata"], user_sequence)
        if final_state == entry["metadata"]["target_value"]:
            return 0.5  # Alternative scoring option

        return 0.1

    def simulate_sequence(self, metadata: dict, sequence: list[str]) -> int:
        """Simulate button presses to verify solutions"""
        state = metadata["initial_value"]
        current_color = metadata["initial_state"]

        buttons = {btn["name"]: btn for btn in metadata["buttons"]}

        for btn_char in sequence:
            btn = buttons.get(btn_char.upper())
            if not btn:
                continue

            # Check button availability
            if btn["active_state"] not in [current_color, "any"]:
                continue

            # Apply operation
            if btn["type"] == "add":
                state += btn["value"]
            elif btn["type"] == "subtract":
                state -= btn["value"]
            elif btn["type"] == "multiply":
                state *= btn["value"]

            # Toggle color state
            current_color = "green" if current_color == "red" else "red"

        return state

    def format_puzzle(self, template, puzzle: dict) -> str:
        return template.format(
            initial_value=puzzle["initial_value"],
            initial_state=puzzle["initial_state"],
            target_value=puzzle["target_value"],
            buttons="\n".join(
                f"{btn['name']}: {btn['type'].title()} {btn['value']} (when {btn['active_state']})"
                for btn in puzzle["buttons"]
            ),
        )


# Register the dataset
register_dataset("quantum_lock", QuantumLockDataset, QuantumLockConfig)

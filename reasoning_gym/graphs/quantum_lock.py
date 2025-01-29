from dataclasses import dataclass
import random
import re
from collections import deque
from typing import List, Optional, Tuple, Dict

from ..factory import ProceduralDataset, register_dataset

@dataclass
class QuantumLockConfig:
    """Configuration for QuantumLock task generation"""

    difficulty: int = 8

class QuantumLockDataset(ProceduralDataset):
    """Generates QuantumLock tasks"""

    def __init__(self, config: QuantumLockConfig):
        self._prompt_templates = ["""\
In front of you are some buttons, a light, and a number. The light will toggle between red and green whenever you press a button. Each button performs a mathematical operation to the number, but the operation may depend on the state of the light.
You must press the shortest correct sequence of buttons to reach the target value.

Start: {initial_value} ({initial_state})
Target: {target_value}
Buttons:
{buttons}"""]
        super().__init__(config=config)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single QuantumLock task

        Returns:
            dict with keys:
                - question: str, the task description
                - answer: str, a solution string
                - metadata: dict with generation parameters
        """

        puzzle_data = self.generate_quantum_puzzle(self.config.difficulty)

        return {
            "question": self.format_puzzle(random.choice(self._prompt_templates), puzzle=puzzle_data),
            "answer": " → ".join(puzzle_data['solution']),
            "metadata": {
                "difficulty": self.config.difficulty,
                "solution_path": puzzle_data['solution'],
                "target_value": puzzle_data['target_value'],
                "buttons": puzzle_data['buttons'],
                "initial_state": puzzle_data['initial_state'],
                "initial_value": puzzle_data['initial_value']
            }
        }

    def generate_quantum_puzzle(self, difficulty=1):
        """
        Generates a Quantum Lock puzzle with configurable difficulty.
        Returns a dictionary containing puzzle parameters and solution.
        """
        # Define possible operations and states
        operations = [
            {'type': 'add', 'values': [1, 2]},
            {'type': 'subtract', 'values': [1, 2]},
            {'type': 'multiply', 'values': [2]}
        ]

        # Generate random buttons
        buttons = []
        for i in range(3):
            op = random.choice(operations)
            btn = {
                'name': chr(65 + i),
                'type': op['type'],
                'value': random.choice(op['values']),
                'active_state': random.choice(['any', 'green'])
            }
            buttons.append(btn)

        # Generate target based on difficulty
        target = random.randint(5 + 5*difficulty, 15 + 10*difficulty)

        # Create puzzle structure
        puzzle = {
            'initial_value': 0,
            'initial_state': 'red',
            'target_value': target,
            'buttons': buttons,
            'max_steps': 8 + 2*difficulty,
            'solution': None
        }

        # Find shortest solution using BFS
        queue = deque([(0, 'red', [])])
        visited = set()

        while queue:
            val, state, path = queue.popleft()

            if val == puzzle['target_value']:
                puzzle['solution'] = path
                return puzzle

            if len(path) >= puzzle['max_steps'] or (val, state) in visited:
                continue

            visited.add((val, state))

            for btn in buttons:
                next_state = 'green' if state == 'red' else 'red'

                # Check if button is usable
                if btn['active_state'] not in [state, 'any']:
                    continue

                # Calculate new value
                try:
                    if btn['type'] == 'add':
                        new_val = val + btn['value']
                    elif btn['type'] == 'subtract':
                        new_val = val - btn['value']
                    elif btn['type'] == 'multiply':
                        new_val = val * btn['value']
                except:
                    continue  # Handle overflows if needed

                queue.append((new_val, next_state, path + [btn['name']]))

        # If no solution found, regenerate
        return self.generate_quantum_puzzle(difficulty)

    def score_answer(self, answer: Optional[str], entry: Dict[str, any]) -> float:
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
            input_str = ''.join(seq) if isinstance(seq, list) else str(seq or "")
            return [c.upper() for c in re.findall(r'[A-C]', input_str.upper())]
        
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

    def simulate_sequence(self, metadata: Dict, sequence: List[str]) -> int:
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
            initial_value=puzzle['initial_value'],
            initial_state=puzzle['initial_state'],
            target_value=puzzle['target_value'],
            buttons='\n'.join(
                f"{btn['name']}: {btn['type'].title()} {btn['value']} (when {btn['active_state']})"
                for btn in puzzle['buttons']
            )
        )

# Register the dataset
register_dataset("QuantumLock", QuantumLockDataset, QuantumLockConfig)

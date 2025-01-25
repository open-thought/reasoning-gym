# Reasoning Gym

We are building a python library of procedural dataset generators and algorithmically verifiable reasoning environments for training Reasoning Models with reinforcement learning (RL).

The goal is to generate virtually infinite data with adjustable complexity.

### How to instantiate a task dataset?

Example:

```python
import reasoning_gym
data = reasoning_gym.create_dataset('leg_counting', size=10, seed=42)
for i, x in enumerate(data):
    print(f'{i}: q="{x['question']}", a="{x['answer']}"')
    print('metadata:', x['metadata'])
```

Output:

```
0: q="How many legs are there in total if you have 1 sea slug, 1 deer?", a="4"
metadata: {'animals': {'sea slug': 1, 'deer': 1}, 'total_legs': 4}
1: q="How many legs are there in total if you have 2 sheeps, 2 dogs?", a="16"
metadata: {'animals': {'sheep': 2, 'dog': 2}, 'total_legs': 16}
2: q="How many legs are there in total if you have 1 crab, 2 lobsters, 1 human, 1 cow, 1 bee?", a="42"
...
```

Available dataset names (which can be used with `create_dataset()`):

```
'polynomial_equations',
'simple_equations',
'base_conversion',
'letter_counting',
'number_filtering',
'number_sorting',
'word_reversal',
'basic_arithmetic',
'chain_sum',
'fraction_simplification',
'gcd',
'lcm',
'leg_counting',
'prime_factorization',
'color_cube_rotation',
'number_sequence',
'countdown',
'maze',
'mini_sudoku',
'sudoku',
'family_relationships',
'propositional_logic',
'syllogism'
```

### Task Overview

#### Algebra Tasks

- `SimpleEquationsDataset`: Generate linear equations with one variable to solve (e.g. "3\*x + 2 = 14")
- `PolynomialEquationsDataset`: Generate polynomial equations with one variable to solve (e.g. "-6*h\*\*4 + 4*h\**2 - 5*h = 0")

#### Arithmetic Tasks

- `BasicArithmeticDataset`: Generate arithmetic expressions with configurable complexity and operators (+, -, \*, /)
- `ChainSum`: Generate addition/subtraction chains with configurable length and digit counts
- `FractionSimplificationDataset`: Generate fraction simplification tasks with configurable complexity
- `GCDDataset`: Generate Greatest Common Divisor problems with configurable number of integers
- `LCMDataset`: Generate Least Common Multiple problems with configurable number of integers
- `LegCountingDataset`: Generate animal leg counting word problems with various animals
- `PrimeFactorizationDataset`: Generate prime factorization tasks with configurable number ranges

#### Algorithmic Tasks

- `BaseConversionDataset`: Convert numbers between different bases (binary, hex, etc.)
- `LetterCountingDataset`: Count letter occurrences in text spans
- `NumberFilteringDataset`: Filter numbers based on comparison with threshold
- `NumberSortingDataset`: Sort lists of numbers in ascending or descending order
- `WordReversalDataset`: Reverse word order in text spans
- `Sorting`

#### Cognition Tasks

- `NumberSequenceDataset`: Generate number sequences with discoverable patterns
- `ColorCubeRotationDataset`: Generate 3D spatial reasoning tasks with colored cube rotations and orientation tracking

#### Logic Tasks

- `PropositionalLogicDataset`: Generate propositional logic reasoning problems

#### Graph Tasks

- `FamilyRelationshipsDataset`: Generate family relationship reasoning tasks with family trees

#### Game Tasks

- `SudokuDataset`: Generate 9x9 Sudoku puzzles with configurable number of empty cells
- `MiniSudokuDataset`: Generate 4x4 Mini Sudoku puzzles with configurable difficulty
- `MazeDataset`: Generates a maze with a start and a goal
- `CountdownDataset`: Generate number game tasks where numbers and operators must be combined to reach a target value

### Available Generators

### PolynomialEquations

Generate polynomial equation with configurable complexity:

```python
from reasoning_gym.algebra import PolynomialEquationsConfig, PolynomialEquationsConfig

config = PolynomialEquationsConfig(
    min_terms=3,
    max_terms=4,
    min_degree=4,
    max_degree=4,
    min_value=1,
    max_value=5,
    size=3,
    seed=123,
)

dataset = PolynomialEquationsDataset(config)
for item in dataset:
    print(item)
```

Example output:

```
{'question': 'Find the real value(s) of b in the equation: b**4 - b**3 - 5*b**2 = 0', 'answer': '[-1.79128784747792, 0.0, 2.79128784747792]', 'metadata': {'polynomial_expr': 'b**4 - b**3 - 5*b**2', 'variable': 'b', 'degree': 4, 'real_solutions': [-1.79128784747792, 0.0, 2.79128784747792]}}
{'question': 'Solve the polynomial equation for real i:\n3*i**4 + 4*i**3 - 1 = 0', 'answer': '[]', 'metadata': {'polynomial_expr': '3*i**4 + 4*i**3 - 1', 'variable': 'i', 'degree': 4, 'real_solutions': []}}
{'question': 'Solve the polynomial equation for real h:\n7*h**4 - 2*h**2 + h = 0', 'answer': '[-0.6998793469266564, 0.0]', 'metadata': {'polynomial_expr': '7*h**4 - 2*h**2 + h', 'variable': 'h', 'degree': 4, 'real_solutions': [-0.6998793469266564, 0.0]}}
```

#### Basic Arithmetic

Generates arithmetic problems with configurable complexity:

```python
from reasoning_gym.arithmetic import BasicArithmeticDataset, BasicArithmeticDatasetConfig

config = BasicArithmeticDatasetConfig(
    min_terms=2,        # Minimum number of terms in expression
    max_terms=4,        # Maximum number of terms
    min_digits=1,       # Minimum digits per number
    max_digits=2,       # Maximum digits per number
    allow_parentheses=True,  # Include nested expressions
    size=5,            # Number of problems to generate
    seed=42            # For reproducibility
)

dataset = BasicArithmeticDataset(config)
for item in dataset:
    print(item)
```

Example output:

```
{'question': '-1 + -5   * 8 + -8 =', 'answer': '-49', 'metadata': {'num_terms': 4, 'num_digits': 1, 'expression': '-1 + -5   * 8 + -8'}}
{'question': '19 - 17 =', 'answer': '2', 'metadata': {'num_terms': 2, 'num_digits': 2, 'expression': '19 - 17'}}
{'question': '3 + -6 * -9 =', 'answer': '57', 'metadata': {'num_terms': 3, 'num_digits': 1, 'expression': '3 + -6 * -9'}}
{'question': '-22 - -94 + -97 =', 'answer': '-25', 'metadata': {'num_terms': 3, 'num_digits': 2, 'expression': '-22 - -94 + -97'}}
{'question': '51 * 63 =', 'answer': '3213', 'metadata': {'num_terms': 2, 'num_digits': 2, 'expression': '51 * 63'}}
```

#### Chain Sum

Generates addition/subtraction problems with configurable complexity:

```python
from reasoning_gym.arithmetic import ChainSum, ChainSumConfig

config = ChainSumConfig(
    min_terms=2,        # Minimum numbers to add/subtract
    max_terms=6,        # Maximum numbers
    min_digits=1,       # Minimum digits per number
    max_digits=4,       # Maximum digits per number
    allow_negation=True, # Allow negative numbers
    size=5,             # Number of problems
    seed=42             # For reproducibility
)

dataset = ChainSum(config)
for item in dataset:
    print(item)
```

Example data:

```
{
    "question": "426 + 562 =",
    "answer": "988",
    "metadata": { "num_terms": 2, "num_digits": 3, "expression": "426 + 562" },
}
{
    "question": "426 + 562 =",
    "answer": "988",
    "metadata": { "num_terms": 2, "num_digits": 3, "expression": "426 + 562" }
}
```

#### Sequence Completion

Generates number sequence completion tasks with dynamic pattern generation:

```python
from reasoning_gym.cognition import NumberSequenceDataset, NumberSequenceConfig

config = NumberSequenceConfig(
    min_terms=4,        # Minimum visible terms
    max_terms=8,        # Maximum visible terms
    min_value=-100,     # Minimum allowed number
    max_value=100,      # Maximum allowed number
    max_complexity=3,   # Maximum operations to combine
    size=5,            # Number of sequences
    seed=42            # For reproducibility
)

dataset = NumberSequenceDataset(config)
for item in dataset:
    print(item)
```

Example data:

```
{
    "question": "3, 6, 12, 24, 48, 96, 192, 384, ?",
    "answer": "768",
    "metadata": {"rule": "double", "complexity": 3, "sequence": [3, 6, 12, 24, 48, 96, 192, 384, 768]},
}
{
    "question": "8, 14, 20, 26, 32, 38, 44, ?",
    "answer": "50",
    "metadata": {"rule": "add 6", "complexity": 1, "sequence": [8, 14, 20, 26, 32, 38, 44, 50]},
}
```

#### Color Cube Rotation

Generates 3D spatial reasoning tasks with cube rotations and color tracking:

```python
from reasoning_gym.cognition import ColorCubeRotationDataset, ColorCubeRotationConfig

config = ColorCubeRotationConfig(
    min_rotations=1,     # Minimum number of rotations
    max_rotations=3,     # Maximum number of rotations
    size=5,             # Number of problems to generate
    seed=42             # For reproducibility
)

dataset = ColorCubeRotationDataset(config)
for item in dataset:
    print(item)
```

Example data:

```
{
    "question": "A cube has:\n- a red top side\n- a blue right side\n- a green front side\n- a yellow left side\n- a white back side\n- an orange bottom side\n\nThe cube is rotated so that the side which was before at the front is now at the top.\nThe cube is rotated so that the side which was before at the right is now at the top.\n\nWhat is now the color of the bottom side of the cube?",
    "answer": "yellow",
    "metadata": {
        "initial_state": {"top": "red", "right": "blue", "front": "green", "left": "yellow", "back": "white", "bottom": "orange"},
        "rotations": ["front", "right"],
        "target_side": "bottom",
        "num_rotations": 2
    }
}
```

#### Propositional Logic

Generates logical reasoning tasks with configurable complexity:

```python
from reasoning_gym.logic import PropositionalLogicDataset, PropositionalLogicConfig

config = PropositionalLogicConfig(
    min_vars=2,         # Minimum number of variables
    max_vars=4,         # Maximum number of variables
    min_statements=2,   # Minimum number of given statements
    max_statements=4,   # Maximum number of statements
    max_complexity=3,   # Maximum operator depth
    size=5,            # Number of problems to generate
    seed=42            # For reproducibility
)

dataset = PropositionalLogicDataset(config)
for item in dataset:
    print(item)
```

Example data:

```
{
    "question": "Given:\n1. R\n2. Q\nWhat can we conclude?",
    "answer": "(P ∨ Q)",
    "metadata": {"premises": ["R", "Q"], "variables": ["P", "Q", "R", "S"], "complexity": 3},
}
{
    "question": "Given:\n1. ((Q → P) ∨ (Q → P))\n2. ((Q ↔ Q) → (P → P))\n3. P\nWhat can we conclude?",
    "answer": "(P → P)",
    "metadata": {
        "premises": ["((Q → P) ∨ (Q → P))", "((Q ↔ Q) → (P → P))", "P"],
        "variables": ["P", "Q"],
        "complexity": 3,
    },
}
```

#### Maze

Generates a maze with configurable difficulty:

```python
from reasoning_gym.games import MazeConfig, MazeDataset

config = MazeConfig(
    min_dist=3,
    max_dist=5,
    min_grid_size=5,
    max_grid_size=5,
    size=2,
    seed=4,
)

dataset = MazeDataset(config)

for item in dataset:
    print()
    print(item["question"])
    print(item)
```

Example data:

```
Navigate from 'd' (start) to '}' (goal):

uuuuu
uCCdu
uCCCu
uu}Cu
uuuuu
Legend: 'u' = Wall, 'C' = Path

{'question': "Navigate from 'd' (start) to '}' (goal):\n\nuuuuu\nuCCdu\nuCCCu\nuu}Cu\nuuuuu\nLegend: 'u' = Wall, 'C' = Path\n", 'answer': '3', 'metadata': {'grid_size': 5, 'grid': ['uuuuu', 'uCCdu', 'uCCCu', 'uu}Cu', 'uuuuu'], 'shortest_path_length': 3, 'start': 'd', 'goal': '}', 'wall': 'u', 'path': 'C'}}

Navigate from 'J' (start) to '_' (goal):

<<<<<
<<J<<
<www<
<<w_<
<<<<<
Legend: '<' = Wall, 'w' = Path

{'question': "Navigate from 'J' (start) to '_' (goal):\n\n<<<<<\n<<J<<\n<www<\n<<w_<\n<<<<<\nLegend: '<' = Wall, 'w' = Path\n", 'answer': '3', 'metadata': {'grid_size': 5, 'grid': ['<<<<<', '<<J<<', '<www<', '<<w_<', '<<<<<'], 'shortest_path_length': 3, 'start': 'J', 'goal': '_', 'wall': '<', 'path': 'w'}}
```

### Future Generator Ideas

- More complex math tasks (algebra, geometry)
- Algorithmic tasks (counting, sorting, re-ordering)
- Logic riddles
- Logic inductive programming tasks
- ARC-AGI synthetic riddles

## Call for Contributions

If you have ideas for additional procedural dataset generators please create an issue here or contact us in the `#arc-agi-2` channel of the [GPU-Mode discord server](https://discord.gg/gpumode).

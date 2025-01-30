# Reasoning Gym

We are building a python library of procedural dataset generators and algorithmically verifiable reasoning environments for training Reasoning Models with reinforcement learning (RL).

The goal is to generate virtually infinite data with adjustable complexity.

Algorithmic verification allows to train on tasks like Rubik‘s cube or [Countdown](<https://en.wikipedia.org/wiki/Countdown_(game_show)#Numbers_Round>) which have many correct solutions.

## Set up for development

1. Clone the project

```
git clone https://github.com/open-thought/reasoning-gym.git
```

2. Create a virtual environment (here we use conda)

```
conda create --name reasoning_gym python=3.11 -y
conda activate reasoning_gym
```

3. Link project and install dependencies

```
pip install -e .
```

4. Install development dependencies

```
pip install -r requirements-dev.txt
```

> NOTE: To consume the APIs in reasoning_gym, just install from pip using the following

```
pip install reasoning-gym
```

## How to instantiate a task dataset?

Example:

```python
import reasoning_gym
data = reasoning_gym.create_dataset('leg_counting', size=10, seed=42)
for i, x in enumerate(data):
    print(f'{i}: q="{x['question']}", a="{x['answer']}"')
    print('metadata:', x['metadata'])
    # use the dataset's `score_answer` method for algorithmic verification
    assert data.score_answer(answer=x['answer'], entry=x) == 1.0
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

See the [Dataset Gallery](GALLERY.md) for a complete list of available datasets with examples.

## Task Overview

### <small>Algebra Tasks</small>

- `SimpleEquationsDataset`: Generate linear equations with one variable to solve (e.g. "3\*x + 2 = 14")
- `PolynomialEquationsDataset`: Generate polynomial equations with one variable to solve (e.g. "-6*h\*\*4 + 4*h\**2 - 5*h = 0")

### <small>Arithmetic Tasks</small>

- `BasicArithmeticDataset`: Generate arithmetic expressions with configurable complexity and operators (+, -, \*, /)
- `ChainSum`: Generate addition/subtraction chains with configurable length and digit counts
- `FractionSimplificationDataset`: Generate fraction simplification tasks with configurable complexity
- `GCDDataset`: Generate Greatest Common Divisor problems with configurable number of integers
- `LCMDataset`: Generate Least Common Multiple problems with configurable number of integers
- `LegCountingDataset`: Generate animal leg counting word problems with various animals
- `PrimeFactorizationDataset`: Generate prime factorization tasks with configurable number ranges

### <small>Algorithmic Tasks</small>

- `BaseConversionDataset`: Convert numbers between different bases (binary, hex, etc.)
- `CaesarCipherDataset`: Encrypt/decrypt text using Caesar cipher with configurable rotation
- `LetterCountingDataset`: Count letter occurrences in text spans
- `NumberFilteringDataset`: Filter numbers based on comparison with threshold
- `NumberSortingDataset`: Sort lists of numbers in ascending or descending order
- `WordSortingDataset`: Sort words in ascending or descending order using ASCII/Unicode ordering
- `LetterJumbleDataset`: Unscramble words that have had their letters randomly jumbled
- `SentenceReorderingDataset`: Reorder sentence after words in it have been randomly shuffled
- `SpellBackwardDataset`: Spell individual words backward (e.g. "sun" -> "nus")
- `WordSequenceReversalDataset`: Reverse word order in text spans
- `WordLadderDataset`: Generate word ladder puzzles where one word is transformed into another by changing one letter at a time

### <small>Code Tasks</small>

- `BFDataset`: Generates BF programs of various difficult, from simple string printing to loops and conditional logic

### <small>Cognition Tasks</small>

- `NumberSequenceDataset`: Generate number sequences with discoverable patterns
- `ColorCubeRotationDataset`: Generate 3D spatial reasoning tasks with colored cube rotations and orientation tracking
- `RubiksCubeDataset`: Generate Rubik's Cube configurations and check correct solutions
- `FigletFontDataset`: Generate random words in different "Figlet" fonts for reasoning about the structure of letters

### <small>Logic Tasks</small>

- `PropositionalLogicDataset`: Generate propositional logic reasoning problems

### <small>Graph Tasks</small>

- `FamilyRelationshipsDataset`: Generate family relationship reasoning tasks with family trees
- `QuantumLockDataset`: Generates puzzles which involve stateful arithmetic and a correct sequence of operations

### <small>Game Tasks</small>

- `SudokuDataset`: Generate 9x9 Sudoku puzzles with configurable number of empty cells
- `MiniSudokuDataset`: Generate 4x4 Mini Sudoku puzzles with configurable difficulty
- `MazeDataset`: Generate a maze with a start and a goal
- `CountdownDataset`: Generate number game tasks where numbers and operators must be combined to reach a target value

## Future Generator Ideas

- More complex math tasks (algebra, geometry)
- Algorithmic tasks (counting, sorting, re-ordering)
- Logic riddles
- Logic inductive programming tasks
- ARC-AGI synthetic riddles

## Call for Contributions

If you have ideas for additional procedural dataset generators please create an issue here or contact us in the `#arc-agi-2` channel of the [GPU-Mode discord server](https://discord.gg/gpumode).

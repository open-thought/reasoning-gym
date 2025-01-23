# Reasoning Gym

We are building a python library of procedural dataset generators and algorithmically verifiable reasoning environments for training Reasoning Models with reinforcement learning (RL).

The goal is to generate virtually infinite data with adjustable complexity.


### Available Generators

#### Basic Arithmetic
Generates arithmetic problems with configurable complexity:
```python
from reasoning_gym.arithmetic import ArithmeticDataset, ArithmeticDatasetConfig

config = ArithmeticDatasetConfig(
    min_terms=2,        # Minimum number of terms in expression
    max_terms=4,        # Maximum number of terms
    min_digits=1,       # Minimum digits per number
    max_digits=2,       # Maximum digits per number
    allow_parentheses=True,  # Include nested expressions
    size=5,            # Number of problems to generate
    seed=42            # For reproducibility
)

dataset = ArithmeticDataset(config)
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

Example output:
```
{'question': '-1 + -2 =', 'answer': '-3', 'metadata': {'num_terms': 2, 'num_digits': 1, 'expression': '-1 + -2'}}
{'question': '426 + 562 =', 'answer': '988', 'metadata': {'num_terms': 2, 'num_digits': 3, 'expression': '426 + 562'}}
{'question': '-4 + 3 + -2 + 0 + -9 =', 'answer': '-12', 'metadata': {'num_terms': 5, 'num_digits': 1, 'expression': '-4 + 3 + -2 + 0 + -9'}}
{'question': '5992 - -1556 + -7316 + -65 =', 'answer': '167', 'metadata': {'num_terms': 4, 'num_digits': 4, 'expression': '5992 - -1556 + -7316 + -65'}}
{'question': '-8690 + 9288 =', 'answer': '598', 'metadata': {'num_terms': 2, 'num_digits': 4, 'expression': '-8690 + 9288'}}
```

#### Sequence Completion
Generates number sequence completion tasks with dynamic pattern generation:
```python
from reasoning_gym.cognition import SequenceDataset, SequenceConfig

config = SequenceConfig(
    min_terms=4,        # Minimum visible terms
    max_terms=8,        # Maximum visible terms
    min_value=-100,     # Minimum allowed number
    max_value=100,      # Maximum allowed number
    max_complexity=3,   # Maximum operations to combine
    size=5,            # Number of sequences
    seed=42            # For reproducibility
)

dataset = SequenceDataset(config)
for item in dataset:
    print(item)
```

Example output:
```
{'question': '3, 6, 12, 24, 48, 96, 192, 384, ?', 'answer': '768', 'metadata': {'rule': 'double', 'complexity': 3, 'sequence': [3, 6, 12, 24, 48, 96, 192, 384, 768]}}
{'question': '8, 14, 20, 26, 32, 38, 44, ?', 'answer': '50', 'metadata': {'rule': 'add 6', 'complexity': 1, 'sequence': [8, 14, 20, 26, 32, 38, 44, 50]}}
{'question': '8, 4, 2, 1, 0, 0, 0, ?', 'answer': '0', 'metadata': {'rule': 'halve', 'complexity': 2, 'sequence': [8, 4, 2, 1, 0, 0, 0, 0]}}
{'question': '-6, 15, -6, 15, ?', 'answer': '-6', 'metadata': {'rule': 'multiply by -1 then add 9', 'complexity': 2, 'sequence': [-6, 15, -6, 15, -6]}}
{'question': '10, 2, -6, -14, -22, -30, ?', 'answer': '-38', 'metadata': {'rule': 'add -8', 'complexity': 1, 'sequence': [10, 2, -6, -14, -22, -30, -38]}}
```

### Future Generator Ideas

- More complex math tasks (algebra, geometry)
- Algorithmic tasks (counting, sorting, re-ordering)
- Logic riddles
- Logic inductive programming tasks
- ARC-AGI synthetic riddles


## Call for Contributions

If you have ideas for additional procedural dataset generators or please create an issue here.

Or contact us in the `#arc-agi-2` channel of the [GPU-Mode discord server](https://discord.gg/gpumode).

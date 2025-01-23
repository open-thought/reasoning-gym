# Reasoning Gym

We are building a python library of procedural dataset generators and algorithmically verifiable reasoning environments for training Reasoning Models with reinforcement learning (RL).

The goal is to generate virtually infinite data with adjustable complexity.

## Quick Start

```python
from reasoning_gym.arithmetic import ChainSum, ChainSumConfig

# Configure a simple arithmetic task generator
config = ChainSumConfig(
    min_terms=2,          # At least 2 numbers per expression
    max_terms=4,          # At most 4 numbers per expression
    min_digits=1,         # Single digit numbers
    max_digits=2,         # Up to 2-digit numbers
    allow_negation=False, # Only positive numbers
    size=5,              # Generate 5 examples
    seed=42              # For reproducibility
)

# Create the dataset
dataset = ChainSum(config)

# Generate some examples
for i in range(len(dataset)):
    item = dataset[i]
    print(f"Question: {item['question']}")
    print(f"Answer: {item['answer']}\n")
```

Example output:
```
Question: 7 + 42 - 15 =
Answer: 34

Question: 91 - 8 =
Answer: 83

Question: 4 + 67 - 12 =
Answer: 59

Question: 28 + 35 =
Answer: 63

Question: 51 - 24 + 7 =
Answer: 34
```


### Generator / Environment Ideas

- math tasks
- algorithmic tasks (counting, sorting, re-ordering, ..)
- logic riddles
- logic inductive programming tasks
- ARC-AGI synthetic riddles


## Call for Contributions

If you have ideas for additional procedural dataset generators or please create an issue here.

Or contact us in the `#arc-agi-2` channel of the [GPU-Mode discord server](https://discord.gg/gpumode).

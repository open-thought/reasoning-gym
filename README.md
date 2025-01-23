# Reasoning Gym

We are building a python library of procedural dataset generators and algorithmically verifiable reasoning environments for training Reasoning Models with reinforcement learning (RL).

The goal is to generate virtually infinite data with adjustable complexity.

## Quick Start

```python
from reasoning_gym.arithmetic import ChainSum, ChainSumConfig

# configure a simple arithmetic task generator
config = ChainSumConfig(
    min_terms=2,
    max_terms=6,
    min_digits=1,
    max_digits=4,
    allow_negation=False, # Only positive numbers
    size=5,               # virtual size of dataset
    seed=42               # deterministic results
)

# create the dataset
dataset = ChainSum(config)

# print some examples
for item in dataset:
    print(item)
```

Example output:
```
{'question': '4 + 3 =', 'answer': '7', 'metadata': {'num_terms': 2, 'num_digits': 1, 'expression': '4 + 3'}}
{'question': '812 + 880 =', 'answer': '1692', 'metadata': {'num_terms': 2, 'num_digits': 3, 'expression': '812 + 880'}}
{'question': '2 + 6 + 3 + 4 + 0 =', 'answer': '15', 'metadata': {'num_terms': 5, 'num_digits': 1, 'expression': '2 + 6 + 3 + 4 + 0'}}
{'question': '8995 - 5221 + 2341 + 5967 =', 'answer': '12082', 'metadata': {'num_terms': 4, 'num_digits': 4, 'expression': '8995 - 5221 + 2341 + 5967'}}
{'question': '1654 + 4744 =', 'answer': '6398', 'metadata': {'num_terms': 2, 'num_digits': 4, 'expression': '1654 + 4744'}}
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

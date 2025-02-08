# Reasoning Gym

We are building a python library of procedural dataset generators and algorithmically verifiable reasoning environments for training reasoning models with reinforcement learning (RL).

The goal is to generate virtually infinite data with adjustable complexity.

Algorithmic verification allows to train on tasks like Rubik‘s cube or [Countdown](<https://en.wikipedia.org/wiki/Countdown_(game_show)#Numbers_Round>) which have many correct solutions.

## Dataset Gallery

In [GALLERY.md](https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md) you find example outputs of all datasets available in reasoning-gym.

## Installation

The `reasoning-gym` package requires Python >= 3.11.

Install via pip:

```
pip install reasoning-gym
```

For development setup see [CONTRIBUTING.md](CONTRIBUTING.md#delevloper-setup).


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

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md).

If you have ideas for dataset generators please create an issue here or contact us in the `#reasoning-gym` channel of the [GPU-Mode discord server](https://discord.gg/gpumode).

[![](https://dcbadge.limes.pink/api/server/gpumode?style=flat)](https://discord.gg/gpumode)

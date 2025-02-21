# 💪🧠 Reasoning Gym

**Reasoning Gym** is a community-created Python library of procedural dataset generators and algorithmically verifiable reasoning environments for training reasoning models with reinforcement learning (RL). The goal is to generate virtually infinite training data with adjustable complexity.

It currently provides **more than 80** tasks over many domains, including but not limited to _algebra_, _arithmetic_, _computation_, _cognition_, _geometry_, _graph theory_, _logic_, and many common _games_.

Some tasks have a single correct answer, while others, such as [Rubik‘s Cube](https://en.wikipedia.org/wiki/Rubik%27s_Cube) and [Countdown](<https://en.wikipedia.org/wiki/Countdown_(game_show)#Numbers_Round>), have many correct solutions. To support this, we provide a standard interface for procedurally verifying solutions.

## 🖼️ Dataset Gallery

In [GALLERY.md](https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md), you can find example outputs of all datasets available in `reasoning-gym`.

## ⬇️ Installation

The `reasoning-gym` package requires Python >= 3.11.

Install the latest published [package from PyPI](https://pypi.org/project/reasoning-gym/) via `pip`:

```
pip install reasoning-gym
```

_Note that this project is currently under active development, and the version published on PyPI may be a few days behind `main`._

## 🛠️ Development

For development setup, see [CONTRIBUTING.md](CONTRIBUTING.md#delevloper-setup).

## ✨ Example Usage

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

## 🔍 Evaluation

Instructions for running the evaluation scripts are provided in [eval/README.md](https://github.com/open-thought/reasoning-gym/blob/main/eval/README.md).

Evaluation results of different reasoning models will be tracked in the [reasoning-gym-eval](https://github.com/open-thought/reasoning-gym-eval) repo.

## 👷 Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md).

If you have ideas for dataset generators please create an issue here or contact us in the `#reasoning-gym` channel of the [GPU-Mode discord server](https://discord.gg/gpumode).

[![](https://dcbadge.limes.pink/api/server/gpumode?style=flat)](https://discord.gg/gpumode)

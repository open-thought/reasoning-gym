<p align="center">
     <!-- title -->
        <h1 align="center"><img src="https://github.com/open-thought/reasoning-gym/raw/main/assets/icon.png" alt="Reasoning Gym Logo" style="vertical-align: bottom;" width="54px" height="40px"> Reasoning Gym</h1>
        <!-- teaser -->
        <p align="center">
            <img src="https://github.com/open-thought/reasoning-gym/raw/main/assets/examples.png" width="800px">
        </p>
        <!-- badges -->
        <p align="center">
            <a href="https://arxiv.org/abs/2505.24760" target="_blank" style="margin-right: 10px;">
                <img src="https://img.shields.io/badge/arXiv-2505.24760-b31b1b.svg?style=for-the-badge" alt="Paper PDF">
            </a>
            <a href="https://discord.gg/gpumode" target="_blank">
                <img src="https://dcbadge.limes.pink/api/server/gpumode?style=for-the-badge" alt="Discord Server">
            </a>
        </p>
</p>

## 🧠 About

**Reasoning Gym** is a community-created Python library of procedural dataset generators and algorithmically verifiable reasoning environments for training reasoning models with reinforcement learning (RL). The goal is to generate virtually infinite training data with adjustable complexity.

It currently provides **more than 100** tasks over many domains, including but not limited to _algebra_, _arithmetic_, _computation_, _cognition_, _geometry_, _graph theory_, _logic_, and many common _games_.

Some tasks have a single correct answer, while others, such as [Rubik‘s Cube](https://en.wikipedia.org/wiki/Rubik%27s_Cube) and [Countdown](<https://en.wikipedia.org/wiki/Countdown_(game_show)#Numbers_Round>), have many correct solutions. To support this, we provide a standard interface for procedurally verifying solutions.

## 🖼️ Dataset Gallery

In [GALLERY.md](https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md), you can find example outputs of all datasets available in `reasoning-gym`.

## ⬇️ Installation

The `reasoning-gym` package requires Python >= 3.10.

Install the latest published [package from PyPI](https://pypi.org/project/reasoning-gym/) via `pip`:

```
pip install reasoning-gym
```

_Note that this project is currently under active development, and the version published on PyPI may be a few days behind `main`._

## ✨ Quickstart

Starting to generate tasks using Reasoning Gym is straightforward:

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

Use keyword arguments to pass task-specific configuration values:

```python
reasoning_gym.create_dataset('leg_counting', size=10, seed=42, max_animals=20)
```

Create a composite dataset containing multiple task types, with optional relative task weightings:

```python
from reasoning_gym.composite import DatasetSpec
specs = [
    # here, leg_counting tasks will make up two thirds of tasks
    DatasetSpec(name='leg_counting', weight=2, config={}),  # default config
    DatasetSpec(name='figlet_font', weight=1, config={"min_word_len": 4, "max_word_len": 6}),  # specify config
]
reasoning_gym.create_dataset('composite', size=10, seed=42, datasets=specs)
```

For the simplest way to get started training models with Reasoning Gym, we recommend using the `verifiers` library, which directly supports RG tasks. See `examples/verifiers` for details. However, RG data can be used with any major RL training framework.

## 🔍 Evaluation

Instructions for running the evaluation scripts are provided in [eval/README.md](https://github.com/open-thought/reasoning-gym/blob/main/eval/README.md).

Evaluation results of different reasoning models will be tracked in the [reasoning-gym-eval](https://github.com/open-thought/reasoning-gym-eval) repo.

## 🤓 Training

The `training/` directory has full details of the training runs we carried out with RG for the paper. In our experiments, we utilise custom Dataset code to dynamically create RG samples at runtime, and to access the RG scoring function for use as a training reward. See `training/README.md` to reproduce our runs.

For a more plug-and-play experience, it may be easier to build a dataset ahead of time. See `scripts/hf_dataset/` for a simple script allowing generation of RG data and conversion to a HuggingFace dataset. To use the script, build your dataset configurations in the YAML. You can find a list of tasks and configurable parameters in [the dataset gallery](GALLERY.md). Then run `save_hf_dataset.py` with desired arguments.

The script will save each dataset entries as a row with `question`, `answer`, and `metadata` columns. The RG scoring functions expect the entry object from each row along with the model response to obtain reward values. Calling the scoring function is therefore simple:

```python
from reasoning_gym import get_score_answer_fn

for entry in dataset:
    model_response = generate_response(entry["question"])
    rg_score_fn = get_score_answer_fn(entry["metadata"]["source_dataset"])
    score = rg_score_fn(model_response, entry)
    # do something with the score...
```

## 👷 Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md).

If you have ideas for dataset generators please create an issue here or contact us in the `#reasoning-gym` channel of the [GPU-Mode discord server](https://discord.gg/gpumode).

[![](https://dcbadge.limes.pink/api/server/gpumode?style=flat)](https://discord.gg/gpumode)


## 🚀 Projects Using Reasoning Gym

Following is a list of awesome projects building on top of Reasoning Gym:
- [Verifiers: Reinforcement Learning with LLMs in Verifiable Environments](https://github.com/willccbb/verifiers)
- [(NVIDIA) ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models](https://arxiv.org/abs/2505.24864)
- [(Nous Research) Atropos - an LLM RL Gym](https://github.com/NousResearch/atropos)
- [(PrimeIntellect) SYNTHETIC-2: a massive open-source reasoning dataset](https://www.primeintellect.ai/blog/synthetic-2)
- [(Gensyn) RL Swarm: a framework for planetary-scale collaborative RL](https://x.com/gensynai/status/1937917790922649669)
- [(Axon RL) GEM: a comprehensive framework for RL environments](https://github.com/axon-rl/gem)
- [(FAIR at Meta) OptimalThinkingBench: Evaluating Over and Underthinking in LLMs](https://github.com/facebookresearch/RAM/tree/main/projects/otb)
- [(Gensyn) Sharing is Caring: Efficient LM Post-Training with Collective RL Experience Sharing](https://arxiv.org/abs/2509.08721v1)
- [(MILA) Self-Evolving Curriculum for LLM Reasoning](https://arxiv.org/abs/2505.14970)
- [(MILA) Recursive Self-Aggregation Unlocks Deep Thinking in Large Language Models](https://rsa-llm.github.io)


## 📝 Citation

If you use this library in your research, please cite the paper:

```bibtex
@misc{stojanovski2025reasoninggymreasoningenvironments,
      title={REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards},
      author={Zafir Stojanovski and Oliver Stanley and Joe Sharratt and Richard Jones and Abdulhakeem Adefioye and Jean Kaddour and Andreas Köpf},
      year={2025},
      eprint={2505.24760},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.24760},
}
```

## ⭐️ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=open-thought/reasoning-gym&type=Date)](https://www.star-history.com/#open-thought/reasoning-gym&Date)

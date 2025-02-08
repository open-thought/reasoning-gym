# Contributing to reasoning-gym

### Delevloper Setup

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


## Procedural Datasets

- We are primarily interested in problems/riddles for which guessing the answer has very little chance of success (good example: multiplying numbers). The problem of tasks with small sets of possible answers (like true/false, multiple-choice) is that RL has to deal with very noisy rewards, which makes it for learning faithful Chain-of-Thoughts.
- Each dataset should come with a configuration class, the dataset class derived from `ProceduralDataset` (see [dataset.py](https://github.com/open-thought/reasoning-gym/blob/main/reasoning_gym/dataset.py)) and unit tests.
- All datasets return dict items with the keys `"question"`, `"answer"` and `"metadata"`. When no single good answer can be given set "answer" to `None`.
- For non-trivial datasets override the `score_answer()` method which returns a numeric value in the range [0, 1] to indicate how close the result is to the actual result.
- take a look at a simple dataset implementation like [chain_sum.py](reasoning_gym/arithmetic/chain_sum.py) and [test_chain_sum.py](https://github.com/open-thought/reasoning-gym/blob/main/tests/test_chain_sum.py). 
- provide clear instructions in the question prompt that would allow an average human to produce an asswer in the correct format.


## Submitting Work - Pull-Requets

We're all working on different parts of reasoning-gym together. To make contributions smoothly we recommend the following:

1.  [Fork this project repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo) and clone it to your local machine. (Read more [About Forks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks))
1.  On a [new branch](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository) in your fork (aka a "feature branch" and not `main`) work on a small focused change that only touches on a few files.
1.  Run `pre-commit` and make sure all files have formatting fixed. This simplifies life for reviewers.
1.  Package up a small bit of work that solves part of the problem
    [into a Pull Request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
    and
    [send it out for review](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review).
1.  If you're lucky, we can merge your change into `main` without any problems.
1.  Merge in your change and move on to a new issue or the second step of your current issue.


### Tips

- To keep your PR clean don't include changes of `GALLERY.md` - the overview file is automatically updated regulary automatically
- install the pre-commit hook via `pre-commit install`
- when using AI coding assistants (cursor, aider, ..) please run `pre-commit run -a` to format all files before committing.



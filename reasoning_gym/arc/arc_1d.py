from dataclasses import dataclass
from random import Random
from typing import Optional

from ..dataset import ProceduralDataset
from ..factory import register_dataset


@dataclass
class Arc1DConfig:
    """Configuration for ARC 1D task generation"""

    min_size: int = 10  # Minimum grid size
    max_size: int = 30  # Maximum grid size
    num_train: int = 3  # Number of training examples
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_size > 0, "min_size must be positive"
        assert self.max_size >= self.min_size, "max_size must be >= min_size"
        assert self.num_train > 0, "num_train must be positive"
        assert self.size > 0, "size must be positive"


class Arc1DDataset(ProceduralDataset):
    """
    Generates ARC 1D tasks by randomly selecting from available task generators

    This dataset is a procedural variant of the 1D-ARC dataset which is described in the paper:
    `LLMs and the Abstraction and Reasoning Corpus:  Successes, Failures, and the Importance
    of Object-based Representations` (https://arxiv.org/abs/2305.18354)

    Ilya Sheprut (optozorax) created rust generators for most of the ARC 1d tasks. For
    reasoning-gym rust tasks were machine-converted to python via Sonnet.

    Ilya's original rust code can be found here: https://github.com/optozorax/arc_1d/
    """

    def __init__(self, config: Arc1DConfig):
        from .arc_1d_tasks import ARC_1D_TASKS

        super().__init__(config=config, seed=config.seed, size=config.size)
        self.ARC_1D_TASKS = ARC_1D_TASKS
        self.task_names = list(ARC_1D_TASKS.keys())

    def __getitem__(self, idx: int) -> dict:
        """Generate a single ARC 1D task with training examples

        Args:
            idx: Index of the item to generate

        Returns:
            dict with keys:
                - question: str, the task description and examples
                - answer: str, the expected output format
                - metadata: dict with generation parameters
        """
        # Create deterministic RNG from base seed and idx
        rng = Random(self.seed + idx)

        # Select random task
        task_name = rng.choice(self.task_names)
        task_func, task_kwargs = self.ARC_1D_TASKS[task_name]

        # Generate training examples
        train_examples = []
        size = rng.randint(self.config.min_size, self.config.max_size)

        for _ in range(self.config.num_train):
            example = None
            while example is None:
                example = task_func(rng, size, **task_kwargs)

            train_examples.append(example)

        # Generate test example
        test_example = None
        while test_example is None:
            test_example = task_func(rng, size, **task_kwargs)

        # Format question
        question = "Find the common rule that maps an input grid to an output grid, given the examples below.\n\n"

        # Add training examples
        for i, example in enumerate(train_examples, 1):
            question += f"Example {i}:\n"
            question += "Input:  " + " ".join(str(x) for x in example["input"]) + "\n"
            question += "Output: " + " ".join(str(x) for x in example["output"]) + "\n\n"

        # Add test input
        question += "Below is a test input grid. Predict the corresponding output grid by applying the rule you found. "
        question += "Describe how you derived the rule and your overall reasoning process in detail before you submit your answer. "
        question += "Your final answer must be placed in <output></output> tags and should be just be the text output grid itself.\n\n"
        question += "Input:\n"
        question += " ".join(str(x) for x in test_example["input"])

        return {
            "question": question,
            "answer": " ".join(str(x) for x in test_example["output"]),
            "metadata": {
                "task_name": task_name,
                "size": size,
                "train_examples": train_examples,
                "test_example": test_example,
            },
        }


# Register the dataset
register_dataset("arc_1d", Arc1DDataset, Arc1DConfig)

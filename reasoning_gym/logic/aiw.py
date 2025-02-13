from dataclasses import dataclass, field
from enum import StrEnum
from random import Random
from string import Template
from typing import List, Optional

from ..factory import ProceduralDataset, register_dataset


class TaskType(StrEnum):
    """Defines the type of task for the Alice in Wonderland dataset."""

    SIBLINGS = "siblings"
    FRIENDS = "friends"
    COLLEAGUES = "colleagues"  # Added colleagues task


@dataclass
class AliceInWonderlandConfig:
    """Configuration options for the Alice in Wonderland dataset.

    Attributes:
        male_names (List[str]): List of male names to use in questions.
        female_names (List[str]): List of female names to use in questions. Must include 'Alice'.
        task_types (List[TaskType]): List of task types to include in dataset.
        seed (Optional[int]): Seed for random number generation.
        size (int): Number of samples in the dataset.
        max_entities (int): Max number of siblings/friends/colleagues in questions.
    """

    male_names: List[str] = field(
        default_factory=lambda: [
            "James",
            "John",
            "Robert",
            "Michael",
            "William",
            "David",
            "Richard",
            "Joseph",
            "Thomas",
            "Charles",
            "Bob",
        ]
    )
    female_names: List[str] = field(
        default_factory=lambda: [
            "Mary",
            "Patricia",
            "Jennifer",
            "Linda",
            "Elizabeth",
            "Barbara",
            "Susan",
            "Jessica",
            "Sarah",
            "Margaret",
            "Alice",
        ]
    )
    task_types: List[TaskType] = field(
        default_factory=lambda: [TaskType.SIBLINGS, TaskType.FRIENDS, TaskType.COLLEAGUES]  # Added Colleagues
    )
    seed: Optional[int] = None
    size: int = 10
    max_entities: int = 6  # Added max_entities

    def validate(self) -> None:
        """Validates the configuration parameters."""
        assert len(self.male_names) > 0, "must provide male names"
        assert len(self.female_names) > 0, "must provide female names"
        assert "Alice" in self.female_names, "'Alice' must be in female names"
        assert len(self.task_types) > 0, "must provide at least one task type"
        assert self.max_entities > 0, "max_entities must be positive"


class AliceInWonderlandDataset(ProceduralDataset):
    """
    A procedural dataset inspired by the "Alice in Wonderland" paper.

    The dataset is inspired by the following paper:
       @inproceedings{nezhurina2024alice,
       title={Alice in Wonderland: Simple Tasks Reveal Severe Generalization and
              Basic Reasoning Deficits in State-Of-the-Art Large Language Models},
       author={Marianna Nezhurina and Lucia Cipolina-Kun and Mehdi Cherti and
              Jenia Jitsev},
       booktitle={NeurIPS 2024 Workshop on Scientific Methods for Understanding
                  Deep Learning},
       year={2024},
       url={https://openreview.net/forum?id=Mkl7dzjYiW}
       }

    """

    def __init__(self, config: AliceInWonderlandConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.templates = {
            TaskType.SIBLINGS: [
                Template(
                    "$female_name has $num_brothers brothers and she also has "
                    "$num_sisters sisters. How many sisters does "
                    "$female_name's brother have?"
                ),
                Template(
                    "$female_name has $num_sisters sisters and she also has "
                    "$num_brothers brothers. How many sisters does "
                    "$male_name's brother have?"
                ),
            ],
            TaskType.FRIENDS: [
                Template(
                    "$female_name has $num_male male friends and she also has "
                    "$num_female female friends. They all are friends with each "
                    "other and have no other friends aside. How many female "
                    "friends does $male_name, a male friend of $female_name, "
                    "have?"
                )
            ],
            TaskType.COLLEAGUES: [  # New colleagues templates
                Template(
                    "$female_name has $num_male_colleagues_alice_circle male colleagues and she also has "
                    "$num_female_colleagues_alice_circle female colleagues. These are all colleagues that $female_name has. "
                    "All these mentioned persons around $female_name are colleagues of each other. "
                    "$male_name has $num_male_colleagues_bob_circle male colleagues "
                    "and $num_female_colleagues_bob_circle female colleagues in total. "
                    "All these mentioned persons around $male_name are colleagues of each other. "
                    "The people in the circle around $male_name do not have "
                    "other colleagues aside - with the only exception of Matilda. "
                    "She is colleague of $male_name and she is also colleague of $female_name, "
                    "being part of $female_name's circle. How many female colleagues does Matilda have?"
                ),
            ],
        }

    def _get_aiw(self, rng: Random) -> dict:
        """Generates a single Alice in Wonderland question.

        Args:
            rng (Random): Random number generator.

        Returns:
            dict: A dictionary containing the generated question, the right answer
                and a description of the example.
        """
        task_type = rng.choice(self.config.task_types)
        female_name = rng.choice(self.config.female_names)
        male_name = rng.choice(self.config.male_names)

        if task_type == TaskType.SIBLINGS:
            num_brothers = rng.randint(1, self.config.max_entities)
            num_sisters = rng.randint(1, self.config.max_entities)

            answer = num_sisters + 1
            template = rng.choice(self.templates[TaskType.SIBLINGS])
            question = template.substitute(
                female_name=female_name,
                male_name=male_name,
                num_brothers=num_brothers,
                num_sisters=num_sisters,
            )
        elif task_type == TaskType.FRIENDS:
            num_male = rng.randint(1, self.config.max_entities)
            num_female = rng.randint(1, self.config.max_entities)

            answer = num_female + 1
            template = rng.choice(self.templates[TaskType.FRIENDS])
            question = template.substitute(
                female_name=female_name,
                male_name=male_name,
                num_male=num_male,
                num_female=num_female,
            )
        elif task_type == TaskType.COLLEAGUES:
            num_male_colleagues_alice_circle = rng.randint(1, self.config.max_entities)
            num_female_colleagues_alice_circle = rng.randint(1, self.config.max_entities)
            num_male_colleagues_bob_circle = rng.randint(1, self.config.max_entities)
            num_female_colleagues_bob_circle = rng.randint(1, self.config.max_entities)

            answer = num_female_colleagues_alice_circle + 1
            template = rng.choice(self.templates[TaskType.COLLEAGUES])
            question = template.substitute(
                female_name=female_name,
                male_name=male_name,
                num_male_colleagues_alice_circle=num_male_colleagues_alice_circle,
                num_female_colleagues_alice_circle=num_female_colleagues_alice_circle,
                num_male_colleagues_bob_circle=num_male_colleagues_bob_circle,
                num_female_colleagues_bob_circle=num_female_colleagues_bob_circle,
            )

        return {"question": question, "answer": answer, "metadata": {"task_type": task_type.value}}

    def __getitem__(self, idx: int) -> dict:
        rng = Random(self.seed + idx)
        return self._get_aiw(rng)


register_dataset("aiw", AliceInWonderlandDataset, AliceInWonderlandConfig)

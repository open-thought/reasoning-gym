"""Leg counting task generator"""

from dataclasses import dataclass
from random import Random
from typing import Optional

from ..factory import ProceduralDataset, register_dataset

ANIMALS = {
    # Animals with 0 legs
    "snake": 0,
    "sea slug": 0,
    "jellyfish": 0,
    "flatworm": 0,
    "leech": 0,
    # Animals with 2 legs
    "chicken": 2,
    "bird": 2,
    "human": 2,
    "duck": 2,
    # Animals with 4 legs
    "dog": 4,
    "cat": 4,
    "cow": 4,
    "horse": 4,
    "lion": 4,
    "elephant": 4,
    "giraffe": 4,
    "tiger": 4,
    "deer": 4,
    "sheep": 4,
    # Animals with 5 legs
    "starfish": 5,
    # Animals with 6 legs
    "insect": 6,
    "ant": 6,
    "butterfly": 6,
    "beetle": 6,
    "bee": 6,
    "wasp": 6,
    "grasshopper": 6,
    "cricket": 6,
    "cockroach": 6,
    "praying mantis": 6,
    "firefly": 6,
    # Animals with 8 legs
    "spider": 8,
    "scorpion": 8,
    # Animals with 10 legs
    "crab": 10,
    "lobster": 10,
    "shrimp": 10,
    # Animals with 14 legs
    "woodlouse": 14,
}

QUESTION_TEMPLATE = """Your task is to count how many legs there are in total when given a list of animals.

Example:
- Input: How many legs are there in total if you have 1 duck, 2 deers, 1 spider, 3 cows?
- Output: 30
- Explanation:
    - Ducks have 2 legs each, so 1 duck has 2 legs.
    - Deers have 4 legs each, so 2 deers have 8 legs.
    - Spiders have 8 legs each, so 1 spider has 8 legs.
    - Cows have 4 legs each, so 3 cows have 12 legs.
    - Therefore, the total number of legs is 2 + 8 + 8 + 12 = 30

Now, how many legs are there in total if you have {animals}?
"""


@dataclass
class LegCountingConfig:
    """Configuration for leg counting task generation"""

    min_animals: int = 3  # Minimum number of animals in problem
    max_animals: int = 10  # Maximum number of animals
    max_instances: int = 15  # Maximum instances of each animal
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_animals > 0, "min_animals must be positive"
        assert self.max_animals >= self.min_animals, "max_animals must be >= min_animals"
        assert self.max_instances > 0, "max_instances must be positive"


class LegCountingDataset(ProceduralDataset):
    """Generates leg counting arithmetic tasks"""

    def __init__(self, config: LegCountingConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def _generate_animals(self, rng: Random) -> dict[str, int]:
        """Generate a random set of animals and their counts"""
        num_types = rng.randint(self.config.min_animals, self.config.max_animals)
        animals = {}

        # Select random animals
        selected_animals = rng.sample(list(ANIMALS.keys()), num_types)
        for animal in selected_animals:
            count = rng.randint(1, self.config.max_instances)
            animals[animal] = count

        return animals

    def __getitem__(self, idx: int) -> dict:
        """Generate a single leg counting task"""
        rng = Random(self.seed + idx)

        # Generate random animals and their counts
        animals = self._generate_animals(rng)

        # Calculate total legs
        total_legs = sum(count * ANIMALS[animal] for animal, count in animals.items())

        # Format animal counts for question
        animal_list = []
        for animal, count in animals.items():
            animal_list.append(f"{count} {animal}{'s' if count > 1 else ''}")

        return {
            "question": QUESTION_TEMPLATE.format(animals=", ".join(animal_list)),
            "answer": str(total_legs),
            "metadata": {
                "difficulty": {
                    "num_animals": len(animals),
                },
                "animals": animals,
                "total_legs": total_legs,
            },
        }


register_dataset("leg_counting", LegCountingDataset, LegCountingConfig)

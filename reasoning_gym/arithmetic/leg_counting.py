"""Leg counting task generator"""
from dataclasses import dataclass
from random import Random
from typing import Dict, Optional

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

@dataclass
class LegCountingConfig:
    """Configuration for leg counting task generation"""
    min_animals: int = 2        # Minimum number of animals in problem
    max_animals: int = 5        # Maximum number of animals
    max_instances: int = 3      # Maximum instances of each animal
    seed: Optional[int] = None
    size: int = 500            # Virtual dataset size

    def validate(self):
        """Validate configuration parameters"""
        assert self.min_animals > 0, "min_animals must be positive"
        assert self.max_animals >= self.min_animals, "max_animals must be >= min_animals"
        assert self.max_instances > 0, "max_instances must be positive"


class LegCountingDataset:
    """Generates leg counting arithmetic tasks"""

    def __init__(self, config: LegCountingConfig):
        self.config = config
        self.config.validate()
        self.seed = config.seed if config.seed is not None else Random().randint(0, 2**32)

    def __len__(self) -> int:
        return self.config.size

    def __iter__(self):
        self._current_idx = 0
        return self

    def __next__(self):
        if self._current_idx >= self.config.size:
            raise StopIteration
        item = self[self._current_idx]
        self._current_idx += 1
        return item

    def _generate_animals(self, rng: Random) -> Dict[str, int]:
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
            
        question = "How many legs are there in total if you have " + ", ".join(animal_list) + "?"
        
        return {
            "question": question,
            "answer": str(total_legs),
            "metadata": {
                "animals": animals,
                "total_legs": total_legs
            }
        }


def leg_counting_dataset(
    min_animals: int = 2,
    max_animals: int = 5,
    max_instances: int = 3,
    seed: Optional[int] = None,
    size: int = 500,
) -> LegCountingDataset:
    """Create a LegCountingDataset with the given configuration."""
    config = LegCountingConfig(
        min_animals=min_animals,
        max_animals=max_animals,
        max_instances=max_instances,
        seed=seed,
        size=size,
    )
    return LegCountingDataset(config)

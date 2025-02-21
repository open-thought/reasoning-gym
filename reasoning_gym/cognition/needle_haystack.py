from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset


@dataclass
class NeedleHaystackConfig:
    """Configuration for NeedleHaystack task generation"""

    num_statements: int = 50
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.num_statements > 0, "num_statements must be greater than 0"
        assert self.num_statements < 168387000, f"num_statements must be less than {168387000}"


def generate_unique_triplets(names: list[str], verbs: list[str], subjects: list[str], n: int, rng) -> dict[str, Any]:
    """
    Generate n unique random triplets (name, verb, subject) without generating the full Cartesian product in memory.

    Each triplet is selected based on a unique index derived from a range of
    total possible combinations. Additionally, one of the generated triplets is
    randomly chosen as the 'needle'.

    Args:
        names (list[str]): List of names.
        verbs (list[str]): List of verbs.
        subjects (list[str]): List of subjects.
        n (int): Number of unique triplets to generate.
        rng (random.Random): A pre-seeded random number generator.

    Returns:
        dict[str, Any]: A dictionary with:
            - "triplets": a list of n unique triplets (tuples of (name, verb, subject)),
            - "needle": one triplet randomly chosen from the list.

    Raises:
        ValueError: If n exceeds the total number of unique triplets possible.
    """
    total_possible = len(names) * len(verbs) * len(subjects)

    # Use a range for memory efficiency and sample n unique indices.
    indices = rng.sample(range(total_possible), n)
    triplets: list[tuple[str, str, str]] = []

    num_verbs = len(verbs)
    num_subjects = len(subjects)

    for idx in indices:
        # Compute the corresponding indices for names, verbs, and subjects.
        name_index = idx // (num_verbs * num_subjects)
        remainder = idx % (num_verbs * num_subjects)
        verb_index = remainder // num_subjects
        subject_index = remainder % num_subjects

        triplet = (names[name_index], verbs[verb_index], subjects[subject_index])
        triplets.append(triplet)

    # Select one random triplet as the needle.
    needle = rng.choice(triplets)
    return {"triplets": triplets, "needle": needle}


class NeedleHaystackDataset(ProceduralDataset):
    """Generates "Needle in a Haystack tasks"""

    def __init__(self, config: NeedleHaystackConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single NeedleHaystack task

        Returns:
            dict with keys:
                - question: str, the task description with cube string
                - answer: None, indicating to use the dynamic evaluator
                - metadata: dict with generation parameters and example solution
        """
        from .needle_data import NAMES, SUBJECTS, VERBS

        rng = Random(self.seed + idx)

        stack = generate_unique_triplets(NAMES, VERBS, SUBJECTS, self.config.num_statements, rng)

        stack_text = ""
        for triplet in stack["triplets"]:
            stack_text = stack_text + f"{triplet[0]} {triplet[1]} {triplet[2]}. "
        question = f"Who {stack['needle'][1]} {stack['needle'][2]}? Reply only with a name."

        full_text = stack_text + "\n" + question

        return {
            "question": full_text,
            "answer": stack["needle"][0],
            "metadata": {"question": question},
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the task.

        Args:
            answer (Optional[str]): The user's answer.
            entry (dict[str, Any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        correct_word = entry["answer"]
        if not answer:
            return 0.0  # No answer given

        # Normalize case
        answer = answer.replace(" ", "").strip().lower()
        correct_word = correct_word.strip().lower()

        if answer == correct_word:
            return 1.0  # Correct!

        return 0.01


# Register the dataset
register_dataset("needle_haystack", NeedleHaystackDataset, NeedleHaystackConfig)

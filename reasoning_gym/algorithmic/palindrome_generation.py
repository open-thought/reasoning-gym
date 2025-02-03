import random
import string
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..factory import ProceduralDataset, register_dataset


@dataclass
class PalindromeConfig:
    """
    Configuration for the palindrome task.

    - min_length: Minimum length of the palindrome.
    - max_length: Maximum length of the palindrome.
    - seed: Optional seed for reproducibility.
    - size: Number of palindrome samples in the virtual dataset.
    """

    min_length: int = 3
    max_length: int = 10
    seed: Optional[int] = None
    size: int = 50

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.min_length >= 1, "min_length must be >= 1"
        assert self.max_length >= self.min_length, "max_length must be >= min_length"


class PalindromeDataset(ProceduralDataset):
    """
    Generates a set of letters that can be assembled into a palindrome.
    """

    def __init__(self, config: PalindromeConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """
        Generate a single palindrome task.

        Returns:
            dict with:
            - "question": Set of letters to form a palindrome.
            - "answer": A correct palindrome.
            - "metadata": Includes letter set and generated palindrome.
        """
        rng = random.Random(self.seed + idx)
        length = rng.randint(self.config.min_length, self.config.max_length)
        letters = self._generate_palindrome_letters(rng, length)
        scrambled_letters = rng.sample(letters, len(letters))  # Scramble the order
        palindrome = self._assemble_palindrome(letters)

        question_str = (
            "Rearrange these letters to form a palindrome. A palindrome is a word, phrase, or sequence that reads the same forward and backward.\n\n"
            "For example, if the letters are: a, a, b — a valid palindrome is: aba.\n\n"
            f"Your letters: {', '.join(scrambled_letters)}\n\n"
            "What palindrome can you form from these letters?"
        )

        return {
            "question": question_str,
            "answer": palindrome,
            "metadata": {
                "letters": scrambled_letters,
                "generated_palindrome": palindrome,
            },
        }

    def _generate_palindrome_letters(self, rng: random.Random, length: int) -> list[str]:
        """Generate a set of letters that can form a palindrome."""
        half_length = length // 2
        letters = rng.choices(string.ascii_lowercase, k=half_length)
        if length % 2 == 1:
            middle_letter = rng.choice(string.ascii_lowercase)
            return letters + [middle_letter] + letters[::-1]
        return letters + letters[::-1]

    def _assemble_palindrome(self, letters: list[str]) -> str:
        """Return the palindrome string from the letter set."""
        return "".join(letters)

    def score_answer(self, answer: Optional[str], metadata: Dict[str, Any]) -> float:
        """Determine if the solution provided is a valid palindrome.
        The answer is expected to be a single string

        Expected behavior:
        - Correct answer (palindrome with only correct letters in the correct quantities) gives 1.0
        - An answer that is a palindrome, but not with the same letters as provided, gives 0.05
        - An answer that is a string, but not a palindrome gives 0.02
        - An empty string gives 0.01.
        - None gives 0.0.
        """
        if answer is None or not isinstance(answer, str):
            return 0.0  # No answer given

        if answer == "":
            return 0.01

        answer = answer.strip().lower()
        expected_letters = metadata["letters"]

        # Check if the answer is a palindrome
        if answer != answer[::-1]:
            return 0.02

        # Check if answer contains the same letters as provided (ignoring order)
        if sorted(answer) != sorted(expected_letters):
            return 0.05

        return 1.0  # Correct solution


register_dataset("palindrome", PalindromeDataset, PalindromeConfig)

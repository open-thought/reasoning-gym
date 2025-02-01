import random
import string
from dataclasses import dataclass
from typing import Optional

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
           f"Rearrange these letters to form a palindrome (a word, phrase, or sequence that remains the same in reverse): {', '.join(scrambled_letters)}\n\n"
            "Example format:\n"
            "racecar\n\n"
            "What is your palindrome?"
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


register_dataset("palindrome", PalindromeDataset, PalindromeConfig)

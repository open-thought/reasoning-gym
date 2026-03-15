"""
Cryptarithm puzzle generator (numbers -> letters approach).

Generates puzzles such that:
   WORD1
 + WORD2
 [+ WORD3]
 ---------
   RESULT
where each letter corresponds to exactly one digit (0..9).
No leading letter can be zero (unless allow_leading_zero=True).
"""

from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "cryptarithm"


def verify_cryptarithm_solution(
    mapping: dict[str, int],
    words_letters: list[str],
    result_letters: str,
    allow_leading_zero: bool,
) -> tuple[bool, str]:
    """Validate if a letter-to-digit mapping satisfies the cryptarithm puzzle constraints.

    Args:
        mapping: Dictionary mapping letters to digits (e.g., {'A': 1, 'B': 2})
        words_letters: List of word strings using letters (e.g., ['ABC', 'DE'])
        result_letters: Result string using letters (e.g., 'FGH')
        allow_leading_zero: Whether leading zeros are allowed

    Returns:
        (is_valid, failure_reason) tuple:
        - is_valid: True if mapping satisfies all constraints
        - failure_reason: String describing why validation failed (empty if valid)
    """
    # Collect all letters used in the puzzle
    all_puzzle_letters = set()
    for word in words_letters:
        all_puzzle_letters.update(word)
    all_puzzle_letters.update(result_letters)

    # Check 1: All letters must be mapped
    mapped_letters = set(mapping.keys())
    if mapped_letters != all_puzzle_letters:
        missing = all_puzzle_letters - mapped_letters
        extra = mapped_letters - all_puzzle_letters
        if missing:
            return False, f"Missing mapping for letter(s): {sorted(missing)}"
        if extra:
            return False, f"Extra letter(s) in mapping: {sorted(extra)}"

    # Check 2: All digits must be valid (0-9)
    for letter, digit in mapping.items():
        if not isinstance(digit, int) or digit < 0 or digit > 9:
            return False, f"Invalid digit for letter {letter}: {digit}"

    # Check 3: Uniqueness constraint - each digit can only be assigned to one letter
    digit_values = list(mapping.values())
    if len(set(digit_values)) != len(digit_values):
        return False, "Duplicate digit assignments detected"

    # Check 4: Leading zero constraint (if not allowed)
    if not allow_leading_zero:
        # Check leading letters of all words
        for word in words_letters:
            if word:  # non-empty word
                leading_letter = word[0]
                if mapping.get(leading_letter) == 0:
                    return False, f"Leading letter '{leading_letter}' cannot map to 0"
        # Check leading letter of result
        if result_letters:
            leading_letter = result_letters[0]
            if mapping.get(leading_letter) == 0:
                return False, f"Leading letter '{leading_letter}' in result cannot map to 0"

    # Check 5: Arithmetic constraint - the sum must be correct
    try:
        # Convert each word from letters to numbers
        word_numbers = []
        for word in words_letters:
            number_str = "".join(str(mapping[letter]) for letter in word)
            word_numbers.append(int(number_str))

        # Convert result from letters to number
        result_number_str = "".join(str(mapping[letter]) for letter in result_letters)
        result_number = int(result_number_str)

        # Check if sum is correct
        computed_sum = sum(word_numbers)
        if computed_sum != result_number:
            return (
                False,
                f"Arithmetic equation not satisfied: {word_numbers} sums to {computed_sum}, expected {result_number}",
            )

    except (KeyError, ValueError) as e:
        return False, f"Error applying mapping: {e}"

    # All checks passed
    return True, ""


@dataclass
class CryptarithmConfig:
    """Configuration for Cryptarithm dataset generation."""

    min_words: int = 2  # Minimum number of addends
    max_words: int = 3  # Maximum number of addends
    allow_leading_zero: bool = False
    seed: Optional[int] = None
    size: int = 500  # Number of puzzle instances to generate

    def validate(self):
        """Validate configuration parameters."""
        assert 2 <= self.min_words <= self.max_words, "min_words must be <= max_words, both >= 2."
        assert self.size > 0, "Dataset size must be positive."


class CryptarithmDataset(ProceduralDataset):
    """
    Generates cryptarithm puzzles by:
      1) Randomly choosing integers for each "addend" (with no leading zero if not allowed),
      2) Summing them,
      3) Mapping distinct digits (0..9) to letters (A..Z),
      4) Formatting the puzzle text.

    This approach guarantees sum correctness and avoids repeated failures.
    """

    def __init__(self, config: CryptarithmConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        rng = Random(self.seed + idx)
        return self._create_single_puzzle(rng, idx)

    def _create_single_puzzle(self, rng: Random, idx: int) -> dict:
        """
        Creates one puzzle with N addends (2..3) plus a result.
        Ensures total distinct digits <= 10.
        """
        # 1) Pick how many addends
        n_words = rng.randint(self.config.min_words, self.config.max_words)

        # 2) For each addend, pick a random length (3..5) and then pick a random integer with that many digits.
        #    If leading zero is disallowed, the first digit is from 1..9.
        word_lengths = [rng.randint(3, 5) for _ in range(n_words)]
        words_numbers = []
        for length in word_lengths:
            if self.config.allow_leading_zero:
                # e.g. random integer in [0, 10^length - 1], then zero-pad to length
                num = rng.randint(0, 10**length - 1)
            else:
                # leading digit is from 1..9, rest are from 0..9
                # e.g. random integer in [10^(length-1), 10^length - 1]
                num = rng.randint(10 ** (length - 1), 10**length - 1)
            words_numbers.append(num)

        # 3) Compute the sum
        total_sum = sum(words_numbers)
        # The sum can have up to (max_length+1) digits, which is normal in cryptarithms.

        # 4) Gather all digits from the addends and the sum
        digits_in_use = set()

        def collect_digits(num: int):
            return set(str(num))

        for wn in words_numbers:
            digits_in_use.update(collect_digits(wn))
        digits_in_use.update(collect_digits(total_sum))

        # If we exceed 10 distinct digits, try again (pick new random numbers).
        # In practice, we can loop until success. But for demonstration, let's do a simple re-pick approach.
        # We'll do a while loop up to some attempts:
        if len(digits_in_use) > 10:
            # Just do a recursion call to pick new numbers, ignoring current picks
            return self._create_single_puzzle(rng)

        # 5) Map each digit to a letter
        #    If no leading zero is allowed, the leading digit of each addend + result must not map to '0'.
        #    Actually, we are generating real numeric values, so there's no scenario of leading "0" for
        #    the addends we enforced (except if allow_leading_zero is True).
        #    For the puzzle's perspective, we simply create a random assignment from {digits_in_use} -> letters.
        #    Then the solver has to figure it out. They don't see the digits, only letters.

        digits_in_use_list = sorted(list(digits_in_use))  # e.g. ['0', '1', '3', '9']
        rng.shuffle(digits_in_use_list)  # shuffle so mapping is random
        letters_pool = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
        rng.shuffle(letters_pool)
        chosen_letters = letters_pool[: len(digits_in_use_list)]

        # digit -> letter mapping
        digit_to_letter = {}
        for d, letter in zip(digits_in_use_list, chosen_letters):
            digit_to_letter[d] = letter

        # If leading-zero is not allowed, we must ensure that the first digit of each addend and the sum
        # does not map to the letter that is assigned to digit '0'. If we see a conflict, we can just re-pick
        # or we can try to swap letters. The simplest is to re-pick for demonstration.
        if not self.config.allow_leading_zero and "0" in digit_to_letter:
            zero_letter = digit_to_letter["0"]
            # Check the first digit of each addend and of the sum
            for wn in words_numbers:
                first_digit = str(wn)[0]
                if digit_to_letter.get(first_digit) == zero_letter:
                    # Conflict => re-generate puzzle
                    return self._create_single_puzzle(rng)
            sum_first_digit = str(total_sum)[0]
            if digit_to_letter.get(sum_first_digit) == zero_letter:
                return self._create_single_puzzle(rng)

        # Now we have a stable digit->letter mapping. Let's create the letter->digit mapping for the answer.
        letter_to_digit = {v: int(k) for k, v in digit_to_letter.items()}

        # 6) Convert each integer to its letter representation
        def int_to_letter_str(num: int) -> str:
            return "".join(digit_to_letter[d] for d in str(num))

        words_letters = [int_to_letter_str(num) for num in words_numbers]
        result_letters = int_to_letter_str(total_sum)

        # 7) Create the puzzle text
        #    We'll do the typical vertical format, with a plus sign before the last addend, dashes, then result
        puzzle_lines = []
        max_width = max(len(w) for w in words_letters + [result_letters])
        for i, wl in enumerate(words_letters):
            if i < len(words_letters) - 1:
                # Right align with spaces, +2 for the "  " prefix
                puzzle_lines.append(f"{wl:>{max_width+2}}")
            else:
                # Right align with spaces, +2 for the "+ " prefix
                puzzle_lines.append(f"+ {wl:>{max_width}}")

        # The line of dashes should match the longest line
        puzzle_lines.append("-" * (max_width + 2))
        # Right align the result
        puzzle_lines.append(f"{result_letters:>{max_width+2}}")

        puzzle_text = "\n".join(puzzle_lines)

        question_str = (
            "Solve this cryptarithm:\n\n"
            f"{puzzle_text}\n\n"
            "Each letter stands for a unique digit (0-9). "
            + (
                "Leading letters may be zero.\n"
                if self.config.allow_leading_zero
                else "No leading letter can be zero.\n"
            )
            + 'Provide a comma separated mapping from letters to digits that satisfies the equation in your final answer. Output format: "A=1,B=2,C=3" (without quotes)\n'
        )

        # 8) Create a human-readable answer, e.g. "A=1,B=0,C=9,..."
        # Note: This is ONE valid solution. Other solutions may exist and are equally valid.
        sorted_letter_keys = sorted(letter_to_digit.keys())
        answer_str = ",".join(f"{letter}={letter_to_digit[letter]}" for letter in sorted_letter_keys)

        # 9) Return the final puzzle item
        return {
            "question": question_str,
            "answer": answer_str,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "allow_leading_zero": self.config.allow_leading_zero,
                "letters": list(letter_to_digit.keys()),
                "word_values": words_numbers,
                "sum_number": total_sum,
                "words_letters": words_letters,
                "result_letters": result_letters,
                "digit_to_letter": digit_to_letter,
                "letter_to_digit": letter_to_digit,
                "difficulty": {
                    "words": (self.config.min_words, self.config.max_words),
                },
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the Cryptarithm task.

        Validates that the provided letter-to-digit mapping satisfies all constraints:
        1. All letters are mapped to unique digits (0-9)
        2. Leading letters don't map to 0 (if allow_leading_zero=False)
        3. The arithmetic equation is satisfied

        The function awards 1.0 for any valid solution (not just the stored solution).
        Multiple valid solutions may exist and are equally acceptable.

        Args:
            answer (Optional[str]): The user's answer in format "A=1,B=2,C=3"
            entry (dict[str, Any]): The dataset entry containing puzzle metadata

        Returns:
            float: 1.0 for valid solution, 0.01 for parseable but invalid, 0.0 for parse error
        """
        if not isinstance(answer, str):
            return 0.0

        # Parse the answer into a letter-to-digit mapping
        try:
            predicted_mapping = {}
            for pair in answer.split(","):
                letter, digit_str = pair.strip().split("=")
                letter = letter.strip()
                predicted_mapping[letter] = int(digit_str.strip())
        except (ValueError, AttributeError):
            return 0.0  # Parse error

        # Extract puzzle constraints from metadata
        words_letters = entry["metadata"]["words_letters"]
        result_letters = entry["metadata"]["result_letters"]
        allow_leading_zero = entry["metadata"].get("allow_leading_zero", False)

        # Validate the solution using the helper function
        is_valid, failure_reason = verify_cryptarithm_solution(
            predicted_mapping, words_letters, result_letters, allow_leading_zero
        )

        if is_valid:
            return 1.0
        else:
            return 0.01  # Parseable but doesn't satisfy constraints


class CryptarithmCurriculum(BaseCurriculum):
    """Curriculum for Cryptarithm puzzles."""

    def __init__(self):
        super().__init__(CryptarithmCurriculum.__name__, CryptarithmConfig)

        # Define the attributes
        self._define_attributes(
            RangeAttributeDefinition(
                name="words",
                levels=[2, 5, 10, 50],
                description="Number of words in the cryptarithm puzzle",
                lower_field_name="min_words",
                upper_field_name="max_words",
                ensure_interval=True,
            )
        )


register_dataset(DATASET_NAME, CryptarithmDataset, CryptarithmConfig, CryptarithmCurriculum)

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

from ..factory import ProceduralDataset, register_dataset

EXAMPLE_CASE = """- Input:
  BASE
+ BALL
------
 GAMES

- Output: B=7, A=4, S=8, E=3, L=5, M=9, G=1
- Explanation:
    * BASE + BALL = GAMES, two 4-digit numbers sum to 5 digits, so G = 1.
    * Units: E + L = S (no carry).
    * Tens: S + L = E + 10 (carry 1). Substitute S = E + L to get E + 2L = E + 10, so L = 5.
    * Since S = E + 5 and S is one digit, E < 5.
    * Hundreds: 2A + 1 = M (with carry).
    * Thousands: 2B = A + 10 (carry makes G = 1). So A = 2B - 10.
    * Try B = 7: Then A = 4 and M = 2(4) + 1 = 9.
    * With E < 5, try E = 3: Then S = 8.
    * Solution: B = 7, A = 4, S = 8, E = 3, L = 5, M = 9, G = 1
    * Verify: BASE (7483) + BALL (7455) = GAMES (14938).
"""


@dataclass
class CryptarithmConfig:
    """Configuration for Cryptarithm dataset generation."""

    min_words: int = 2  # Minimum number of addends
    max_words: int = 3  # Maximum number of addends
    allow_leading_zero: bool = False
    include_example: bool = True
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
        return self._create_single_puzzle(rng)

    def _create_single_puzzle(self, rng: Random) -> dict:
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
        if self.config.include_example:
            question_str += "\nHere's an example:\n" + EXAMPLE_CASE

        # 8) Create a human-readable answer, e.g. "A=1,B=0,C=9,..."
        sorted_letter_keys = sorted(letter_to_digit.keys())
        answer_str = ",".join(f"{letter}={letter_to_digit[letter]}" for letter in sorted_letter_keys)

        # 9) Return the final puzzle item
        return {
            "question": question_str,
            "answer": answer_str,
            "metadata": {
                "letters": list(letter_to_digit.keys()),
                "word_values": words_numbers,
                "sum_number": total_sum,
                "words_letters": words_letters,
                "result_letters": result_letters,
                "digit_to_letter": digit_to_letter,
                "letter_to_digit": letter_to_digit,
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the Cryptarithm task.

        The function awards 1.0 for a correct format and answers for all alphabet pairs.

        Args:
            answer (Optional[str]): The user's answer already parsed by `extract_answer`
            answer_str (dict[str, Any]): The original dataset answer_str containing the correct answer. ie "A=1,B=3..."

        Returns:
            float: The computed score between 0.0 and 1.0.
        """
        correct_mapping = {}
        correct_answer_str = entry["answer"]  # e.g. "A=1,B=7,..."
        for pair in correct_answer_str.split(","):
            alphabet, number = pair.split("=")
            correct_mapping[alphabet] = int(number)

        # case 1 : pairs are in a list format and the number of pairs matched up
        if len(answer.split(",")) != len(correct_mapping):
            return 0.1

        predict_mapping = {}
        for pair in answer.split(","):
            try:
                alphabet, number = pair.strip().split("=")
                # as the unique alphabet grows we may want this to scale linearly with the number alphabet
                predict_mapping[alphabet] = int(number)
            except ValueError:
                return 0.15
        # case 2 : all alphabet has correct format ALPHABET=NUMBER format
        if len(predict_mapping) != len(correct_mapping):
            return 0.3

        # case 3 : partial score for the number of correct mapping answer
        total_correct, total = 0, 0
        for alphabet, number in correct_mapping.items():
            total += 1
            if alphabet in predict_mapping:
                if predict_mapping[alphabet] == number:
                    total_correct += 1

        # note: linear relationship is probably not good?
        return (total_correct / total) * 0.7 + 0.3


register_dataset("cryptarithm", CryptarithmDataset, CryptarithmConfig)

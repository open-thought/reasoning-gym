"""Caesar cipher task generator"""

from dataclasses import dataclass
from random import Random
from typing import Optional

from reasoning_gym.data import read_data_file

from ..factory import ProceduralDataset, register_dataset


@dataclass
class CaesarCipherConfig:
    """Configuration for Caesar cipher task generation"""

    delimiter: str = "."  # Delimiter for splitting text into sentences
    min_words: int = 3  # Minimum words per sentence
    max_words: int = 20  # Maximum words per sentence
    min_rotation: int = 1  # Minimum Caesar rotation
    max_rotation: int = 25  # Maximum Caesar rotation
    seed: Optional[int] = None
    size: int = 500  # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_words > 0, "min_words must be positive"
        assert self.max_words >= self.min_words, "max_words must be >= min_words"
        assert 0 < self.min_rotation <= self.max_rotation < 26, "rotation must be in range [1,25]"


class CaesarCipherDataset(ProceduralDataset):
    """Generates Caesar cipher encryption/decryption tasks"""

    def __init__(self, config: CaesarCipherConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

        # Load and preprocess text
        text = read_data_file("in_the_year_2889.txt")

        # Split into sentences and filter
        sentences = [s.strip() for s in text.split(config.delimiter) if s.strip()]

        # Process each sentence
        self.valid_sentences = []
        for sentence in sentences:
            # Split into words and filter for alpha-only
            words = [w.upper() for w in sentence.split() if w.isalpha()]
            if self.config.min_words <= len(words) <= self.config.max_words:
                self.valid_sentences.append(" ".join(words))

    def _caesar_encrypt(self, text: str, rotation: int) -> str:
        """Apply Caesar cipher encryption with given rotation"""
        result = []
        for char in text:
            if char.isalpha():
                # Convert to 0-25 range, rotate, convert back to ASCII
                base = ord("A")
                rotated = (ord(char) - base + rotation) % 26
                result.append(chr(base + rotated))
            else:
                result.append(char)
        return "".join(result)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single Caesar cipher task"""
        rng = Random(self.seed + idx)

        # Select random sentence and rotation
        sentence = rng.choice(self.valid_sentences)
        rotation = rng.randint(self.config.min_rotation, self.config.max_rotation)

        # Generate cipher text
        cipher_text = self._caesar_encrypt(sentence, rotation)

        return {
            "question": f"Decrypt this Caesar cipher text: {cipher_text}. Provide only the decrypted text as your final answer.",
            "answer": sentence,
            "metadata": {"rotation": rotation, "cipher_text": cipher_text, "clear_text": sentence},
        }


register_dataset("caesar_cipher", CaesarCipherDataset, CaesarCipherConfig)

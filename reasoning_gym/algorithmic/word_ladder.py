"""Word ladder task generator"""

from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..data import get_data_file_path
from ..factory import ProceduralDataset, register_dataset

QUESTION_TEMPLATE = """Transform the word ladder '{start}' to '{end}' by changing one letter at a time.
Provide your answer as a comma-separated sequence of uppercase letters without spaces.
Each step must be a valid English word."""


@dataclass
class WordLadderConfig:
    """Configuration for word ladder task generation"""

    min_word_length: int = 4  # Minimum word length
    max_word_length: int = 4  # Maximum word length
    min_chain_length: int = -1  # Set to -1 for shortest path or a minimum of 3
    max_chain_length: int = -1  # Set to -1 for shortest path or a max
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_word_length >= 3, "min_word_length must be >= 3"
        assert self.max_word_length >= self.min_word_length, "max_word_length must be >= min_word_length"
        assert self.max_word_length <= 5, "max_word_length must be <= 5"

        # Add size validation
        if self.size > 20000:  # Add reasonable upper limit
            raise ValueError("Dataset size too large for this algorithm and constraints")

        # Modified validation logic
        if self.min_chain_length == -1:
            if self.max_chain_length != -1:
                assert (
                    self.max_chain_length >= 3
                ), "When min_chain_length=-1 (shortest path), max_chain_length must be -1 or >=3"
        elif self.max_chain_length == -1:
            raise AssertionError("max_chain_length cannot be -1 unless min_chain_length is also -1")
        else:
            assert self.min_chain_length >= 3, "min_chain_length must be 3 or -1"
            assert self.max_chain_length >= self.min_chain_length, "max_chain_length must be >= min_chain_length"

    def is_valid_path_length(self, length: int) -> bool:
        """Check if a path length meets the configuration requirements"""
        # When min_chain_length is -1, we accept any path of length >= 3
        if self.min_chain_length == -1:
            if self.max_chain_length == -1:
                return length >= 3
            return 3 <= length <= self.max_chain_length

        # Otherwise check against both min and max
        return (
            self.min_chain_length <= length <= (self.max_chain_length if self.max_chain_length != -1 else float("inf"))
        )


class WordLadderDataset(ProceduralDataset):
    """Generates word ladder transformation tasks"""

    def __init__(self, config: WordLadderConfig):
        self.config = config
        self.word_sets = {}
        self.word_graphs = {}
        self._vocabulary = None  # A large list of dictionary words to validate words against

        # Load words from CSV
        self.word_sets = self._load_words_from_csv(
            min_length=self.config.min_word_length, max_length=self.config.max_word_length
        )

        # Precompute word graphs for all lengths
        for length in range(self.config.min_word_length, self.config.max_word_length + 1):
            self.word_graphs[length] = self._build_word_graph(length)

        config.validate()
        super().__init__(config=config, seed=config.seed, size=config.size)

    @classmethod
    def _load_words_from_csv(cls, min_length: int = 3, max_length: int = 5) -> dict[int, set[str]]:
        """Load words from CSV file organized by length"""
        # Validate length range before processing
        assert 3 <= min_length <= max_length <= 5, "Word length must be between 3 and 5 inclusive"

        import csv

        word_sets = {}

        try:
            # Get CSV content as string
            with get_data_file_path("words.csv").open("r", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)

                for row in reader:
                    # Process each word length column using config range
                    for length in range(min_length, max_length + 1):
                        col_name = f"{length}_letter"
                        word = row.get(col_name, "")

                        if not word:  # Skip empty entries
                            continue

                        word_sets.setdefault(length, set()).add(word.upper())

        except Exception as e:
            raise RuntimeError(f"Error processing words.csv content: {e}") from e

        # Validate we have words for each length
        for length in range(min_length, max_length + 1):
            if length not in word_sets or not word_sets[length]:
                raise ValueError(f"No valid words found for length {length}")

        return word_sets

    def _get_neighbors(self, word: str, word_set: set[str]) -> set[str]:
        """Get neighbors from either precomputed graph or by computing on demand"""
        # Try precomputed graph first
        if len(word) in self.word_graphs and word in self.word_graphs[len(word)]:
            return self.word_graphs[len(word)].get(word, set())

        # Fall back to computing neighbors directly for custom word sets
        neighbors = set()
        for i in range(len(word)):
            for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                neighbor = word[:i] + c + word[i + 1 :]
                if neighbor != word and neighbor in word_set:
                    neighbors.add(neighbor)
        return neighbors

    def _build_word_graph(self, word_length: int) -> dict[str, set[str]]:
        """Build graph of word connections for given length, using caching"""
        # Return cached graph if it exists
        if word_length in self.word_graphs:
            return self.word_graphs[word_length]

        # Build new graph
        word_set = self.word_sets[word_length]
        graph = {}

        # Build connections
        for word in word_set:
            neighbors = set()
            for i in range(word_length):
                for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    neighbor = word[:i] + c + word[i + 1 :]
                    if neighbor != word and neighbor in word_set:
                        neighbors.add(neighbor)
            graph[word] = neighbors

        # Cache and return
        self.word_graphs[word_length] = graph
        return self.word_graphs[word_length]

    def _find_path(self, start: str, end: str, word_set: set[str]) -> Optional[list[str]]:
        """Simplified path finding using BFS for shortest paths"""
        # Early exit if words are direct neighbors
        if end in self._get_neighbors(start, word_set):
            return [start, end]

        # Use basic BFS for shortest path
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()
            if current == end:
                if self.config.is_valid_path_length(len(path)):
                    return path
                return None

            for neighbor in self._get_neighbors(current, word_set):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))

        return None

    def _generate_word_pair(self, rng: Random, length: int) -> tuple[str, str, list[str]]:
        """Simplified word pair generation"""
        word_set = self.word_sets[length]
        words_list = sorted(word_set)
        max_attempts = 100

        for _ in range(max_attempts):
            start = rng.choice(words_list)
            end = rng.choice(words_list)

            if start == end:
                continue

            path = self._find_path(start, end, word_set)
            if path:
                return start, end, path

        raise RuntimeError(f"Failed to find valid pair for length {length}")

    def __getitem__(self, idx: int) -> dict:
        """Generate a single word ladder task"""
        if idx >= self.size:
            raise IndexError(f"Dataset index {idx} out of range for size {self.size}")

        try:
            rng = Random(self.seed + idx)
            length = rng.randint(self.config.min_word_length, self.config.max_word_length)
            start, end, path = self._generate_word_pair(rng, length)
        except RuntimeError as e:
            # If we run out of valid paths, adjust the virtual size
            self.size = idx
            raise IndexError(f"Dataset exhausted at index {idx}. {str(e)}")

        return {
            "question": QUESTION_TEMPLATE.format(start=start, end=end),
            "answer": ",".join(path),
            "metadata": {"start_word": start, "end_word": end, "word_length": length, "chain_length": len(path)},
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        if answer is None:
            return 0

        answer_words = tuple(s.strip() for s in answer.upper().split(","))

        metadata = entry["metadata"]
        start_word = metadata["start_word"]
        end_word = metadata["end_word"]
        word_length = len(end_word)
        known_words = self.word_sets[word_length]

        # Check conditions:
        # 1. start and end word match question
        # 2. all words have the correct length
        # 3. every changed word is a single letter change from the previous word
        # 4. all words are in our vocabulary

        if len(answer_words) < 2:
            return 0

        if answer_words[0] != start_word or answer_words[-1] != end_word:
            return 0.01

        if not all(len(w) == word_length for w in answer_words):
            return 0.01

        for i in range(1, len(answer_words)):
            if sum(1 for a, b in zip(answer_words[i - 1], answer_words[i]) if a != b) != 1:
                return 0.01

        reward = 1.0
        for word in answer_words:
            if not word in known_words:
                reward *= 0.5

        return reward


register_dataset("word_ladder", WordLadderDataset, WordLadderConfig)

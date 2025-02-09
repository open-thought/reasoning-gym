"""Word ladder task generator"""

from collections import deque
from dataclasses import dataclass
from random import Random
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path

from reasoning_gym.data import read_data_file, read_json_file

from ..factory import ProceduralDataset, register_dataset


@dataclass
class WordLadderConfig:
    """Configuration for word ladder task generation"""

    min_word_length: int = 4  # Minimum word length
    max_word_length: int = 4  # Maximum word length
    min_chain_length: int = -1  # Set to -1 for shortest path or a minimum of 3
    max_chain_length: int = -1  # Set to -1 for shortest path or a max
    seed: Optional[int] = None
    size: int = 500
    dictionary_file_path: str = "words_dictionary.json"

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
        self._word_dict = None # A large list of dictionary words to validate words against

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
    def _load_words_from_csv(cls, min_length: int = 3, max_length: int = 5) -> Dict[int, Set[str]]:
        """Load words from CSV file organized by length"""
        # Validate length range before processing
        assert 3 <= min_length <= max_length <= 5, "Word length must be between 3 and 5 inclusive"

        import csv
        from io import StringIO

        word_sets = {}

        try:
            # Get CSV content as string
            csv_content = read_data_file("words.csv")

            # Use StringIO to create a file-like object from the string
            csv_file = StringIO(csv_content)
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
    
    def _load_word_dictionary(self, file_path: str) -> Dict[str, Any]:
        """Load word dictionary from JSON file"""
        return read_json_file(file_path)
    
    @property
    def word_dict(self) -> Dict[str, Any]:
        """Lazy loading of word dictionary"""
        if self._word_dict is None:
            self._word_dict = self._load_word_dictionary(self.config.dictionary_file_path)
        return self._word_dict
    
    # Lazy loading of word dictionary
    def load_word_dictionary(self, file_path: str) -> Set[str]:
        if not hasattr(self, "_word_dict"):
            self._word_dict = self._load_word_dictionary(file_path)
        return self._word_dict

    def _get_neighbors(self, word: str, word_set: Set[str]) -> Set[str]:
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

    def _build_word_graph(self, word_length: int) -> Dict[str, Set[str]]:
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

    def _find_path(self, start: str, end: str, word_set: Set[str]) -> Optional[List[str]]:
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

    def _generate_word_pair(self, rng: Random, length: int) -> Tuple[str, str, List[str]]:
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
            "question": f"Transform the word ladder '{start}' to '{end}' by changing one letter at a time.",
            "answer": ",".join(path),
            "metadata": {"start_word": start, "end_word": end, "word_length": length, "chain_length": len(path)},
        }
    
    def score_answer(self, answer: Optional[str], entry: Dict[str, any]) -> float:
        oracle_answer = entry["answer"].upper().strip()
        answer = answer.upper().strip() if answer is not None else None
        is_answer_correct = set(answer.split(",")) == set(oracle_answer.split(","))
        word_dict = self.word_dict

        # NOTE: I am assuming that answer is a comma-separated string of words and that if it exactly matches the oracle answer
        # it is correct and gets a reward of 1.0.

        # Check for two conditions:
        # 1. Ensure all words in the answer are valid (Assuming all generated words would be found in the word_dict)
        # 2. Ensure every changed word is a single letter change from the previous word
        is_all_words_valid = all(word in word_dict for word in answer.split(","))
        words = answer.split(",")
        total_words = len(words)
        single_letter_change_words = 0
        for i in range(1, len(words)):
            if sum(1 for a, b in zip(words[i - 1], words[i]) if a != b) == 1:
                single_letter_change_words += 1
        # Number of comparisons should be total_words - 1
        is_all_single_letter_change = single_letter_change_words == (total_words - 1)

        reward = 0.0
        if answer is not None:
            if is_answer_correct:
                reward = 1.0
            elif is_all_words_valid:
                reward = 0.5
            elif is_all_single_letter_change:
                reward = 0.5
            elif single_letter_change_words > 0:
                reward = single_letter_change_words / (total_words - 1)
            else:
                reward = 0.01

        return reward


register_dataset("word_ladder", WordLadderDataset, WordLadderConfig)

"""Word ladder task generator"""

from dataclasses import dataclass
from random import Random
from typing import List, Optional, Set, Dict, Tuple
from collections import deque
from reasoning_gym.data import read_data_file

from ..factory import ProceduralDataset, register_dataset

@dataclass
class WordLadderConfig:
    """Configuration for word ladder task generation"""
    
    min_word_length: int = 3       # Minimum word length
    max_word_length: int = 5       # Maximum word length
    min_chain_length: int = 3      # Minimum transformations required
    max_chain_length: int = 11     # Maximum transformations required
    seed: Optional[int] = None
    size: int = 500                # Virtual dataset size

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.min_word_length > 2, "min_word_length must be 3"
        assert self.max_word_length >= self.min_word_length, "max_word_length must be >= min_word_length"
        assert self.max_word_length <= 5, "max_word_length must be 5"
        assert self.min_chain_length > 2, "min_chain_length must be 3"
        assert self.max_chain_length >= self.min_chain_length, "max_chain_length must be >= min_chain_length"

class WordLadderDataset(ProceduralDataset):
    """Generates word ladder transformation tasks"""

    def __init__(self, config: WordLadderConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        
        # Load words from CSV file
        self.word_sets = self._load_words_from_csv()

    def _load_words_from_csv(self) -> Dict[int, Set[str]]:
        """Load words from CSV file organized by length"""
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
                # Process each word length column
                for length in range(3, 6):
                    col_name = f'{length}_letter'
                    word = row.get(col_name, '')
                    
                    if not word:  # Skip empty entries
                        continue
                        
                    if self.config.min_word_length <= length <= self.config.max_word_length:
                        word_sets.setdefault(length, set()).add(word.upper())
                        
        except Exception as e:
            raise RuntimeError(f"Error processing words.csv content: {e}") from e
        
        # Validate we have words for each length
        for length in range(self.config.min_word_length, self.config.max_word_length + 1):
            if length not in word_sets or not word_sets[length]:
                raise ValueError(f"No valid words found for length {length}")
                
        return word_sets

    def _differs_by_one(self, word1: str, word2: str) -> bool:
        """Check if two words differ by exactly one letter"""
        if len(word1) != len(word2):
            return False
        differences = 0
        for c1, c2 in zip(word1, word2):
            if c1 != c2:
                differences += 1
                if differences > 1:
                    return False
        return differences == 1

    def _find_path(self, start: str, end: str, word_set: Set[str]) -> Optional[List[str]]:
        """Find shortest path between start and end words using BFS"""
        if start == end:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            # Try changing one letter at a time
            word_chars = list(current)
            for i in range(len(word_chars)):
                for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    if word_chars[i] == c:
                        continue
                        
                    # Create new word
                    word_chars[i] = c
                    new_word = ''.join(word_chars)
                    
                    # Check if it's a valid word and not visited
                    if new_word in word_set and new_word not in visited:
                        new_path = path + [new_word]
                        if new_word == end:
                            return new_path
                        
                        queue.append((new_word, new_path))
                        visited.add(new_word)
                
                # Restore original character
                word_chars[i] = current[i]
        
        return None

    def _generate_word_pair(self, rng: Random, length: int) -> Tuple[str, str, List[str]]:
        """Generate valid start/end words with solution path"""
        word_set = self.word_sets[length]
        max_attempts = 500
        
        for _ in range(max_attempts):
            start, end = rng.sample(sorted(word_set), 2)
            path = self._find_path(start, end, word_set)
            if path and self.config.min_chain_length <= len(path) <= self.config.max_chain_length:
                return start, end, path
        
        raise RuntimeError(f"Failed to find valid pair for length {length} after {max_attempts} attempts")

    def __getitem__(self, idx: int) -> dict:
        """Generate a single word ladder task"""
        rng = Random(self.seed + idx)
        length = rng.randint(self.config.min_word_length, self.config.max_word_length)
        start, end, path = self._generate_word_pair(rng, length)
        
        return {
            "question": f"Transform the word '{start}' into '{end}' by changing one letter at a time. Each step must create a valid English word (including plurals) and keep the same word length. Show the sequence of words needed.",
            "answer": ",".join(path),
            "metadata": {
                "start_word": start,
                "end_word": end,
                "word_length": length,
                "chain_length": len(path)
            }
        }


register_dataset("word_ladder", WordLadderDataset, WordLadderConfig)
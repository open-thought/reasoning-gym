"""Coaching module for difficulty adjustment and score tracking"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .dataset import ProceduralDataset


@dataclass
class ScoreBoard:
    """Tracks scores and metadata for coaching sessions"""
    
    scores: List[float] = field(default_factory=list)
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    conversations: List[Optional[List[Dict]]] = field(default_factory=list)

    def add_score(self, score: float, metadata: Dict[str, Any], conversation: Optional[List[Dict]] = None) -> None:
        """Add a new score entry with associated metadata and optional conversation

        Args:
            score: The score achieved (typically 0.0-1.0)
            metadata: Dictionary of metadata about the task/attempt
            conversation: Optional list of conversation turns as dicts
        """
        self.scores.append(score)
        self.metadata.append(metadata)
        self.conversations.append(conversation)


class Coach(ProceduralDataset):
    """A dataset wrapper that tracks performance and adjusts difficulty
    
    The Coach wraps a ProceduralDataset (typically a CompositeDataset) and:
    1. Tracks scores and metadata in a ScoreBoard
    2. Adjusts difficulty based on performance (to be implemented)
    """

    def __init__(self, dataset: ProceduralDataset):
        """Initialize with inner dataset
        
        Args:
            dataset: The ProceduralDataset to wrap
        """
        super().__init__(config=dataset.config, seed=dataset.seed, size=dataset.size)
        self.dataset = dataset
        self.score_board = ScoreBoard()

    def __getitem__(self, idx: int) -> dict:
        """Forward item generation to inner dataset"""
        return self.dataset[idx]

    def score_answer(self, answer: Optional[str], entry: Dict[str, Any], 
                    conversation: Optional[List[Dict]] = None) -> float:
        """Score answer and track results
        
        Args:
            answer: The answer to score
            entry: The task entry containing question/answer/metadata
            conversation: Optional conversation history as list of message dicts

        Returns:
            float: Score between 0.0 and 1.0
        """
        # Get score from inner dataset
        score = self.dataset.score_answer(answer, entry)
        
        # Track score and metadata
        self.score_board.add_score(
            score=score,
            metadata=entry["metadata"],
            conversation=conversation
        )
        
        # Update difficulty based on recent performance
        self.update_difficulty()
        
        return score

    def update_difficulty(self) -> None:
        """Update difficulty based on recent performance
        
        To be implemented in future versions.
        """
        pass  # Placeholder for future difficulty adjustment logic

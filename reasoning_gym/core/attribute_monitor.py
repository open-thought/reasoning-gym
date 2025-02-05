from typing import Dict, List, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

@dataclass
class AttributeMonitor:
    """Monitors performance for a specific attribute."""
    window_size: int = 10  # Number of recent problems to track
    warmup_count: int = 10  # Number of problems before starting analysis # TODO: Implement warmup (not just level_history as can go back to level)
    degradation_threshold: float = 0.9  # Threshold for degradation
    std_plateau_threshold: float = 0.1  # Threshold for plateau
    
    def __post_init__(self):
        self.curriculum = None  # Will be set during initialization
        self.attribute_name = None  # Will be set during initialization
        self.recent_scores: List[float] = []  # List of recent accuracy scores
        self.level_history: Dict[int, List[float]] = defaultdict(list)  # Scores for each difficulty level
        self.best_scores: Dict[int, float] = {}  # Best smoothed score achieved at each level
    
    def initialize(self, curriculum: Any, attribute_name: str):
        """Initialize monitor with curriculum and attribute."""
        self.curriculum = curriculum
        self.attribute_name = attribute_name
        self.set_level(curriculum.get_attr_level(attribute_name))
    
    @property
    def current_level(self) -> int:
        """Get current level from curriculum."""
        return self.curriculum.get_attr_level(self.attribute_name)
    
    def increment_level(self) -> bool:
        """Increment difficulty level using curriculum."""
        if self.curriculum.increment_attr_level(self.attribute_name):
            self.recent_scores = []  # Reset scores for new level
            return True
        return False
    
    def decrement_level(self) -> bool:
        """Decrement difficulty level using curriculum."""
        if self.curriculum.decrement_attr_level(self.attribute_name):
            self.recent_scores = []  # Reset scores for new level
            return True
        return False
    
    def set_level(self, level: int):
        """Set difficulty level using curriculum."""
        self.curriculum.set_attr_level(self.attribute_name, level)
        self.recent_scores = []  # Reset scores for new level
    
    def add_score(self, score: float):
        """Add a new score and update metrics."""
        self.recent_scores.append(score)
        if len(self.recent_scores) > self.window_size:
            self.recent_scores.pop(0)
        
        self.level_history[self.current_level].append(score)
        
        # Update best score if current moving average is higher
        if len(self.recent_scores) >= self.window_size:
            current_avg = np.mean(self.recent_scores)
            self.best_scores[self.current_level] = max(
                current_avg,
                self.best_scores.get(self.current_level, float('-inf'))
            )
    
    def get_current_accuracy(self) -> float:
        """Get the current moving average accuracy."""
        if len(self.recent_scores) < self.window_size:
            return 0.0
        return np.mean(self.recent_scores)
    
    # TODO: is_*, addscore merge
    def is_improving(self) -> bool:
        """Check if performance is improving."""
        if len(self.recent_scores) < self.window_size:
            return False
        current_avg = np.mean(self.recent_scores)
        return current_avg > self.best_scores.get(self.current_level, float('-inf'))
    
    def is_plateau(self) -> bool:
        """Check if performance has plateaued."""
        if len(self.recent_scores) < self.window_size:
            return False
            
        recent_std = np.std(self.recent_scores)
        return recent_std < self.std_plateau_threshold
    
    def is_degrading(self) -> bool:
        """Check if performance is degrading."""
        if len(self.recent_scores) < self.window_size:
            return False
            
        current_avg = np.mean(self.recent_scores)
        return current_avg < self.best_scores.get(self.current_level, float('-inf')) * self.degradation_threshold
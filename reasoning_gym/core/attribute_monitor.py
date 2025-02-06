from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from enum import Enum

class PerformanceTrend(Enum):
    """Performance trend states for an attribute."""
    INSUFFICIENT_DATA = "insufficient_data"
    IMPROVING = "improving"
    PLATEAU_HIGH_ACC = "plateau_high_acc" 
    PLATEAU_LOW_ACC = "plateau_low_acc"
    DEGRADING = "degrading"
    STABLE = "stable"

@dataclass
class AttributeMonitor:
    """Monitors performance for a specific attribute."""
    # TODO: Different vars for different exercises, attributes
    window_size: int = 10  # Number of recent problems to track
    warmup_count: int = 10  # Number of problems before starting analysis # TODO: Implement warmup (not just level_history as can go back to level)
    high_acc_threshold: float = 0.8  # Threshold for high accuracy
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

    def add_score(self, score: float) -> PerformanceTrend:
        """
        Add a new score and analyze the performance trend.

        Returns:
            PerformanceTrend: The current performance trend
        """
        self.recent_scores.append(score)
        if len(self.recent_scores) > self.window_size:
            self.recent_scores.pop(0)

        self.level_history[self.current_level].append(score)

        # Not enough data to analyze trends
        if len(self.recent_scores) < self.window_size:
            return PerformanceTrend.INSUFFICIENT_DATA

        current_avg = np.mean(self.recent_scores)
        current_best = self.best_scores.get(self.current_level, float('-inf'))

        # Update best score if current moving average is higher
        if current_avg > current_best:
            self.best_scores[self.current_level] = current_avg
            return PerformanceTrend.IMPROVING

        # Check for plateau
        recent_std = np.std(self.recent_scores)
        if recent_std < self.std_plateau_threshold:
            if current_avg > self.high_acc_threshold:
                return PerformanceTrend.PLATEAU_HIGH_ACC
            else:
                return PerformanceTrend.PLATEAU_LOW_ACC

        # Check for degradation
        if current_avg < current_best * self.degradation_threshold:
            return PerformanceTrend.DEGRADING

        return PerformanceTrend.STABLE

    def get_current_accuracy(self) -> float:
        """Get the current moving average accuracy."""
        if len(self.recent_scores) < self.window_size:
            return 0.0
        return np.mean(self.recent_scores)
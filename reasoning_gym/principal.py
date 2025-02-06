from typing import Dict, Optional, Any
from enum import Enum
import logging
from collections import defaultdict
import numpy as np
from reasoning_gym.core.attribute_monitor import AttributeMonitor
from reasoning_gym.core.exercise_registrar import ExerciseRegistrar
from reasoning_gym.core.attribute_monitor import PerformanceTrend

class CurriculumMode(Enum):
    """Mode of curriculum operation for the Principal."""
    PREDEFINED = "predefined"  # Follow a pre-defined curriculum
    DYNAMIC = "dynamic"        # Dynamically adjust based on performance

class Principal:
    """Manages exercise difficulty and curriculum progression."""

    def __init__(self, mode: CurriculumMode = CurriculumMode.DYNAMIC):
        self.exercises = {}  # type: Dict[str, Any]  # Exercise instances
        self.exercise_curricula = {}  # type: Dict[str, Any]  # Loaded curricula
        self.current_levels = defaultdict(dict)  # Current difficulty levels
        self.performance_monitors = defaultdict(dict)  # Attribute monitors
        self.curriculum_mode = mode
        self.logger = logging.getLogger(__name__)

        # Auto-register exercises
        registered = ExerciseRegistrar.register_all()
        for exercise_name, (exercise, curriculum) in registered.items():
            self.register_exercise(exercise_name, exercise, curriculum)

    def register_exercise(self, exercise_name: str, exercise_instance: Any, 
                         curriculum: Any) -> None:
        """Register a new exercise with its curriculum."""
        self.exercises[exercise_name] = exercise_instance
        self.exercise_curricula[exercise_name] = curriculum

        # Initialize monitors for each attribute
        for attr_name, attr_def in curriculum.attributes.items():
            monitor = AttributeMonitor()
            monitor.initialize(curriculum, attr_name)
            self.performance_monitors[exercise_name][attr_name] = monitor

        self.logger.info(f"Principal: Registered exercise: {exercise_name} with {len(curriculum.attributes)} attributes")

    def generate_problem(self, exercise_name: str) -> tuple:
        """Generate a problem from the specified exercise."""
        if exercise_name not in self.exercises:
            raise KeyError(f"Principal: Exercise {exercise_name} not registered")

        exercise = self.exercises[exercise_name]
        problem = exercise.generate()
        return problem

    # TODO:Implement predefined
    def update_performance(self, exercise_name: str, attribute_name: str, 
                          score: float) -> None:
        """Update performance metrics for an attribute."""

        monitor = self.performance_monitors[exercise_name][attribute_name]
        atrr_trend =monitor.add_score(score)

        if self.curriculum_mode == CurriculumMode.DYNAMIC:
            self._adjust_difficulty(exercise_name, attribute_name, trend=atrr_trend)

    # TODO: Implement representation
    def _adjust_difficulty(self, exercise_name: str, attribute_name: str, trend: Optional[PerformanceTrend] = None) -> None:
        """Adjust difficulty based on performance metrics."""
        monitor = self.performance_monitors[exercise_name][attribute_name]

        # TODO: If plateau and < threshold or degrading, increase representation, if persists n steps, decrease difficulty
        match trend:
            case PerformanceTrend.IMPROVING:
                # Keep current level while improving
                return
            case PerformanceTrend.PLATEAU_HIGH_ACC:
                # Try to increase difficulty if accuracy is high
                if monitor.increment_level():
                    self.logger.info(
                        f"Principal: Increasing difficulty for {exercise_name}.{attribute_name} "
                        f"to level {monitor.current_level}"
                    )
            case PerformanceTrend.DEGRADING:
                # If performance is degrading, decrease difficulty
                if monitor.decrement_level():
                    self.logger.info(
                        f"Principal: Decreasing difficulty for {exercise_name}.{attribute_name} "
                        f"to level {monitor.current_level}"
                    )
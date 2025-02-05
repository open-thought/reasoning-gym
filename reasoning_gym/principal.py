from typing import Dict, Optional, Any
from enum import Enum
import logging
from collections import defaultdict
import numpy as np
from reasoning_gym.core.attribute_monitor import AttributeMonitor

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
        self.plateau_threshold = 0.8
        self.logger = logging.getLogger(__name__)
        
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
            
        self.logger.info(f"Registered exercise: {exercise_name}")
        
    def generate_problem(self, exercise_name: str) -> tuple:
        """Generate a problem from the specified exercise."""
        if exercise_name not in self.exercises:
            raise KeyError(f"Exercise {exercise_name} not registered")
            
        exercise = self.exercises[exercise_name]
        
        # Set current attribute levels before generation
        for attr_name, monitor in self.performance_monitors[exercise_name].items():
            exercise.set_attribute_level(attr_name, monitor.current_level)
            
        return exercise.generate()
    
    # TODO:Implement predefined
    def update_performance(self, exercise_name: str, attribute_name: str, 
                          score: float) -> None:
        """Update performance metrics for an attribute."""
            
        monitor = self.performance_monitors[exercise_name][attribute_name]
        monitor.add_score(score)
        
        if self.curriculum_mode == CurriculumMode.DYNAMIC:
            self._adjust_difficulty(exercise_name, attribute_name)
        
    # TODO: Implement representation
    def _adjust_difficulty(self, exercise_name: str, attribute_name: str) -> None:
        """Adjust difficulty based on performance metrics."""
        monitor = self.performance_monitors[exercise_name][attribute_name]
        
        # Implementation of the adjustment logic
        if monitor.is_improving():
            # Keep current level while improving
            return
        elif monitor.is_plateau():
            current_acc = monitor.get_current_accuracy()
            # Try to increase difficulty if accuracy is high
            if current_acc > self.plateau_threshold:
                if monitor.increment_level():
                    self.logger.info(
                        f"Increasing difficulty for {exercise_name}.{attribute_name} "
                        f"to level {monitor.current_level}"
                    )
        elif monitor.is_degrading():
            # If performance is degrading, decrease difficulty
            if monitor.decrement_level():
                self.logger.info(
                    f"Decreasing difficulty for {exercise_name}.{attribute_name} "
                    f"to level {monitor.current_level}"
                )
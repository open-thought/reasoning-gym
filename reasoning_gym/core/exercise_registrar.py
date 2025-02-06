import logging
from typing import Dict, Any, Tuple
from reasoning_gym import exercises, curricula

logger = logging.getLogger(__name__) # TODO: Why only here

class ExerciseRegistrar:
    """Handles registration of exercises and curricula."""

    @staticmethod
    def register_all() -> Dict[str, Tuple[Any, Any]]:
        """
        Register all exercises and their curricula.
        Returns dict of {exercise_name: (exercise_instance, curriculum_instance)}.
        """
        registered = {}

        # Get all Dataset classes from exercises module
        for exercise_name in exercises.__all__:
            if exercise_name.endswith('Dataset'):
                exercise_class = getattr(exercises, exercise_name)
                exercise_base = exercise_name[:-7]  # Remove 'Dataset'
                curriculum_name = f"{exercise_base}Curriculum"

                if hasattr(curricula, curriculum_name):
                    try:
                        curriculum_class = getattr(curricula, curriculum_name)

                        # Create instances
                        exercise_instance = exercise_class()
                        curriculum_instance = curriculum_class()

                        # Convert CamelCase to snake_case for exercise name
                        exercise_name = ''.join([f'_{c.lower()}' if c.isupper() else c
                                               for c in exercise_base]).lstrip('_')

                        registered[exercise_name] = (exercise_instance, curriculum_instance)
                        logger.info(f"ExerciseRegistrar: Registered exercise: {exercise_name}")
                    except Exception as e:
                        logger.error(f"ExerciseRegistrar: Error instantiating {exercise_name}: {e}", exc_info=True)
                else:
                    logger.warning(f"ExerciseRegistrar: No curriculum found for {exercise_name}")

        return registered
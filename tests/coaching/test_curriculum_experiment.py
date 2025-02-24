import pytest

from reasoning_gym.coaching.curriculum_config import CurriculumAttributeConfig, CurriculumExperimentConfig
from reasoning_gym.coaching.experiment import CurriculumExperiment


def test_curriculum_experiment_initialization():
    """Test basic initialization of CurriculumExperiment"""

    # Create config with leg_counting dataset
    config = CurriculumExperimentConfig(
        curricula={"leg_counting": CurriculumAttributeConfig(attribute_levels={"num_animals": 2}, weight=1.0)}
    )

    # Create experiment
    experiment = CurriculumExperiment(name="test_experiment", config=config, size=10, seed=42)

    # Check experiment was created correctly
    assert experiment.name == "test_experiment"
    assert "leg_counting" in experiment.curricula
    assert "leg_counting" in experiment.composite.datasets

    # Check curriculum was configured correctly
    curriculum = experiment.curricula["leg_counting"]
    assert curriculum.get_attr_level("num_animals") == 2

    # Check dataset was created with correct config
    dataset = experiment.composite.datasets["leg_counting"]
    assert dataset.config.min_animals == 1
    assert dataset.config.max_animals == 3

    # Check we can get entries from the dataset
    entry = experiment.get_dataset_entry(0)
    assert "question" in entry
    assert "answer" in entry
    assert "metadata" in entry
    assert entry["metadata"]["source_dataset"] == "leg_counting"


def test_curriculum_experiment_wildcard_level():
    """Test using "*" to set all attribute levels"""

    config = CurriculumExperimentConfig(
        curricula={"leg_counting": CurriculumAttributeConfig(attribute_levels={"*": 3}, weight=1.0)}
    )

    experiment = CurriculumExperiment(name="test_experiment", config=config, size=10, seed=42)

    # Check all attributes were set to level 3
    curriculum = experiment.curricula["leg_counting"]
    for attr_name in curriculum.attributes:
        assert curriculum.get_attr_level(attr_name) == 3


def test_curriculum_experiment_mixed_levels():
    """Test mixing "*" with specific attribute levels"""

    config = CurriculumExperimentConfig(
        curricula={
            "leg_counting": CurriculumAttributeConfig(
                attribute_levels={"*": 2, "num_animals": 4}, weight=1.0  # Should override the "*" level
            )
        }
    )

    experiment = CurriculumExperiment(name="test_experiment", config=config, size=10, seed=42)

    curriculum = experiment.curricula["leg_counting"]
    assert curriculum.get_attr_level("num_animals") == 4  # Specific override

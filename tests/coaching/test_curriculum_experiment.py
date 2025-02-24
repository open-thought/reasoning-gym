import os
import tempfile
import pytest
import yaml

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
    
    
def test_curriculum_experiment_from_yaml():
    """Test loading curriculum experiment config from YAML"""
    
    # Create a temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_content = """
        curricula:
          leg_counting:
            attribute_levels:
              "*": 2
              num_animals: 4
            weight: 1.5
          caesar_cipher:
            attribute_levels:
              difficulty: 3
            weight: 0.8
        """
        f.write(yaml_content)
        yaml_path = f.name
    
    try:
        # Load config from YAML
        config = CurriculumExperimentConfig.from_yaml(yaml_path)
        
        # Verify config was loaded correctly
        assert len(config.curricula) == 2
        assert "leg_counting" in config.curricula
        assert "caesar_cipher" in config.curricula
        
        # Check leg_counting curriculum
        leg_counting = config.curricula["leg_counting"]
        assert leg_counting.attribute_levels["*"] == 2
        assert leg_counting.attribute_levels["num_animals"] == 4
        assert leg_counting.weight == 1.5
        
        # Check caesar_cipher curriculum
        caesar_cipher = config.curricula["caesar_cipher"]
        assert caesar_cipher.attribute_levels["difficulty"] == 3
        assert caesar_cipher.weight == 0.8
        
        # Create experiment from the loaded config
        experiment = CurriculumExperiment(name="yaml_test", config=config, size=10, seed=42)
        
        # Verify experiment was created correctly
        assert "leg_counting" in experiment.curricula
        assert "caesar_cipher" in experiment.curricula
        
        # Check attribute levels were applied
        leg_curriculum = experiment.curricula["leg_counting"]
        assert leg_curriculum.get_attr_level("num_animals") == 4
        
        caesar_curriculum = experiment.curricula["caesar_cipher"]
        assert caesar_curriculum.get_attr_level("difficulty") == 3
        
    finally:
        # Clean up the temporary file
        os.unlink(yaml_path)

import os

import pytest
import yaml

from reasoning_gym.composite import CompositeConfig, CompositeDataset, DatasetSpec
from reasoning_gym.version_manager import DatasetVersionManager


def create_test_config(tmp_path):
    """Create a test YAML config file"""
    config = {
        "size": 100,
        "seed": 42,
        "datasets": [
            {
                "name": "chain_sum",
                "weight": 0.3,
                "config": {
                    "min_terms": 2,
                    "max_terms": 4,
                },
            },
            {
                "name": "leg_counting",
                "weight": 0.7,
                "config": {
                    "min_animals": 1,
                    "max_animals": 3,
                },
            },
        ],
    }

    config_path = os.path.join(tmp_path, "test_config.yaml")
    print(config_path)
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


def test_composite_config_validation():
    """Test configuration validation"""
    with pytest.raises(AssertionError):
        config = CompositeConfig(size=-1)
        config.validate()

    with pytest.raises(AssertionError):
        config = CompositeConfig(datasets=[])
        config.validate()


def test_composite_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = CompositeConfig(
        size=10, seed=42, datasets=[DatasetSpec("chain_sum", 1.0, {"min_terms": 2, "max_terms": 4})]
    )

    dataset1 = CompositeDataset(config)
    dataset2 = CompositeDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_composite_dataset_metadata():
    """Test that metadata includes source dataset information"""
    config = CompositeConfig(
        size=10, seed=42, datasets=[DatasetSpec("chain_sum", 1.0, {"min_terms": 2, "max_terms": 4})]
    )

    dataset = CompositeDataset(config)
    item = dataset[0]

    assert "source_dataset" in item["metadata"]
    assert "source_index" in item["metadata"]
    assert item["metadata"]["source_dataset"] == "chain_sum"
    assert isinstance(item["metadata"]["source_index"], int)


def test_composite_dataset_weights():
    """Test that dataset weights are properly normalized"""
    config = CompositeConfig(
        size=1000,
        seed=42,
        datasets=[
            DatasetSpec("chain_sum", 2.0, {"min_terms": 2}),
            DatasetSpec("products", 3.0, {"min_terms": 2}),
        ],
    )

    dataset = CompositeDataset(config)
    assert abs(dataset.weights[0] - 2.0) < 1e-6
    assert abs(dataset.weights[1] - 3.0) < 1e-6

    # Test weight updates
    dataset.update_dataset_weight("chain_sum", 1.0)
    print(dataset.weights)
    assert abs(dataset.weights[0] - 1.0) < 1e-6
    assert abs(dataset.weights[1] - 3.0) < 1e-6

    # Test invalid weight
    with pytest.raises(ValueError, match="Weight must be non-negative"):
        dataset.update_dataset_weight("chain_sum", -1.0)

    # Test invalid dataset name
    with pytest.raises(KeyError):
        dataset.update_dataset_weight("invalid_dataset", 1.0)

    # Test zero total weight
    dataset.update_dataset_weight("chain_sum", 0.0)
    with pytest.raises(ValueError, match="Total of weights must be greater than zero"):
        dataset.update_dataset_weight("products", 0.0)
        _ = dataset[0]  # access item with all weights 0

    # Test duplicate dataset names
    with pytest.raises(ValueError, match="Duplicate dataset names"):
        CompositeConfig(
            size=1000,
            seed=42,
            datasets=[
                DatasetSpec("chain_sum", 1.0, {"min_terms": 2}),
                DatasetSpec("chain_sum", 1.0, {"min_terms": 3}),
            ],
        ).validate()


def test_version_tracking_with_config_updates():
    """Test that version tracking works correctly when updating dataset configs"""
    # Create composite dataset with version manager
    version_manager = DatasetVersionManager()
    config = CompositeConfig(
        size=10, seed=42, datasets=[DatasetSpec("chain_sum", 1.0, {"min_terms": 2, "max_terms": 4})]
    )
    dataset = CompositeDataset(config, version_manager=version_manager)

    # Get an entry and its id from initial version
    entry_1 = dataset[0]
    entry_id_1 = entry_1["metadata"]["entry_id"]
    answer_1 = entry_1["answer"]

    # Update dataset config
    dataset.update_dataset_config("chain_sum", {"min_terms": 3, "max_terms": 5})

    # Get new entry after config update
    entry_2 = dataset[0]
    entry_id_2 = entry_2["metadata"]["entry_id"]
    answer_2 = entry_2["answer"]

    # Verify entries have different version IDs
    version_1 = int(entry_id_1.split(".")[0])
    version_2 = int(entry_id_2.split(".")[0])
    assert version_1 != version_2, "New config should create new version"

    # Verify original answer still works with original version
    score_1 = dataset.score_answer_with_id(answer_1, entry_id_1)
    assert score_1 == 1.0, "Original answer should still work with original version"

    # Verify new answer works with new version
    score_2 = dataset.score_answer_with_id(answer_2, entry_id_2)
    assert score_2 == 1.0, "New answer should work with new version"

    # Verify original answer fails with new version
    score_3 = dataset.score_answer_with_id(answer_1, entry_id_2)
    assert score_3 < 1.0, "Original answer should not work with new version"


def test_score_answer_with_id():
    """Test scoring answers using entry_id"""
    # Create composite dataset with version manager
    version_manager = DatasetVersionManager()
    config = CompositeConfig(
        size=10, seed=42, datasets=[DatasetSpec("chain_sum", 1.0, {"min_terms": 2, "max_terms": 4})]
    )
    dataset = CompositeDataset(config, version_manager=version_manager)

    # Get an entry and its id
    entry = dataset[0]
    entry_id = entry["metadata"]["entry_id"]

    # Test successful scoring
    answer = entry["answer"]
    score = dataset.score_answer_with_id(answer, entry_id)
    assert score == 1.0  # Correct answer should get full score

    # Test wrong answer
    wrong_answer = "wrong"
    score = dataset.score_answer_with_id(wrong_answer, entry_id)
    assert score < 1.0  # Wrong answer should get lower score

    # Test invalid entry_id format
    with pytest.raises(ValueError, match="Invalid entry_id format"):
        dataset.score_answer_with_id(answer, "invalid")

    # Test non-existent version
    with pytest.raises(KeyError, match="Version .* not found"):
        dataset.score_answer_with_id(answer, "999.0")

    # Test without version manager
    dataset_no_vm = CompositeDataset(config)
    with pytest.raises(RuntimeError, match="Version manager required"):
        dataset_no_vm.score_answer_with_id(answer, entry_id)


def test_add_remove_dataset():
    """Test adding and removing datasets from composite"""
    config = CompositeConfig(
        size=1000,
        seed=42,
        datasets=[
            DatasetSpec("chain_sum", 1.0, {"min_terms": 2}),
        ],
    )

    dataset = CompositeDataset(config)

    # Test adding new dataset
    new_spec = DatasetSpec("products", 2.0, {"min_terms": 2})
    dataset.add_dataset(new_spec)

    assert len(dataset.datasets) == 2
    assert "products" in dataset.datasets
    assert len(dataset.config.datasets) == 2

    assert dataset.dataset_names[0] == "chain_sum"
    assert dataset.dataset_names[1] == "products"
    assert abs(dataset.weights[0] - 1.0) < 1e-6  # chain_sum weight
    assert abs(dataset.weights[1] - 2.0) < 1e-6  # products weight

    # Test duplicate name
    with pytest.raises(ValueError, match="already exists"):
        dataset.add_dataset(new_spec)

    # Test removing dataset
    dataset.remove_dataset("products")
    assert len(dataset.datasets) == 1
    assert "products" not in dataset.datasets
    assert len(dataset.config.datasets) == 1

    # Test removing non-existent dataset
    with pytest.raises(KeyError):
        dataset.remove_dataset("nonexistent")

    # Test removing last dataset
    with pytest.raises(ValueError, match="Cannot remove last dataset"):
        dataset.remove_dataset("chain_sum")


def test_yaml_loading(tmp_path):
    """Test loading configuration from YAML"""
    config_path = create_test_config(tmp_path)
    config = CompositeConfig.from_yaml(config_path)

    assert config.size == 100
    assert config.seed == 42
    assert len(config.datasets) == 2
    assert config.datasets[0].name == "chain_sum"
    assert config.datasets[1].name == "leg_counting"

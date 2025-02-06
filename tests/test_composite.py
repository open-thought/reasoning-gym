import os

import pytest
import yaml

from reasoning_gym.composite import CompositeConfig, CompositeDataset, DatasetSpec


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
            DatasetSpec("chain_sum", 3.0, {"min_terms": 3}),
        ],
    )

    dataset = CompositeDataset(config)
    assert abs(dataset.weights[0] - 0.4) < 1e-6
    assert abs(dataset.weights[1] - 0.6) < 1e-6


def test_yaml_loading(tmp_path):
    """Test loading configuration from YAML"""
    config_path = create_test_config(tmp_path)
    config = CompositeConfig.from_yaml(config_path)

    assert config.size == 100
    assert config.seed == 42
    assert len(config.datasets) == 2
    assert config.datasets[0].name == "chain_sum"
    assert config.datasets[1].name == "leg_counting"

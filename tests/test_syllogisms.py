"""Tests for syllogism task generation"""

import pytest

from reasoning_gym.logic.syllogisms import Quantifier, SyllogismConfig, SyllogismDataset, Term


def test_syllogism_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = SyllogismConfig(
            allow_all=False,
            allow_no=False,
            allow_some=False,
            allow_some_not=False,
        )  # No quantifiers allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = SyllogismConfig(invalid_ratio=-0.1)  # Invalid ratio
        config.validate()

    with pytest.raises(AssertionError):
        config = SyllogismConfig(invalid_ratio=1.1)  # Invalid ratio
        config.validate()


def test_syllogism_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = SyllogismConfig(seed=42, size=10)
    dataset1 = SyllogismDataset(config)
    dataset2 = SyllogismDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_syllogism_dataset_items():
    """Test basic properties of generated items"""
    config = SyllogismConfig(size=10, seed=42)
    dataset = SyllogismDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "premise1" in item["metadata"]
        assert "premise2" in item["metadata"]
        assert "conclusion" in item["metadata"]
        assert "is_valid" in item["metadata"]

        # Verify answer format
        assert item["answer"] in ("Yes", "No")

        # Verify question format
        assert "Consider these statements:" in item["question"]
        assert "1." in item["question"]
        assert "2." in item["question"]
        assert "Does it logically follow that:" in item["question"]


def test_syllogism_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = SyllogismConfig(size=5, seed=42)
    dataset = SyllogismDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_syllogism_custom_terms():
    """Test syllogism generation with custom terms"""
    custom_terms = [
        Term("programmer", "programmers"),
        Term("coder", "coders"),
        Term("developer", "developers"),
    ]
    config = SyllogismConfig(terms=custom_terms, size=10, seed=42)
    dataset = SyllogismDataset(config)

    for item in dataset:
        # Verify only custom terms are used
        text = item["question"] + str(item["metadata"])
        assert any(term.name in text or term.plural in text for term in custom_terms)
        # Verify default terms are not used
        assert "mortal" not in text
        assert "human" not in text


def test_syllogism_validity():
    """Test logical validity rules"""
    config = SyllogismConfig(
        allow_all=True,
        allow_no=False,
        allow_some=False,
        allow_some_not=False,
        include_invalid=False,  # Only generate valid syllogisms
        size=10,
        seed=42,
    )
    dataset = SyllogismDataset(config)

    for item in dataset:
        # All valid ALL syllogisms should have "Yes" as answer
        assert item["answer"] == "Yes"
        assert item["metadata"]["is_valid"] is True

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


def test_valid_syllogism_forms():
    """Test specific valid syllogistic forms"""
    config = SyllogismConfig(size=1, seed=42)
    dataset = SyllogismDataset(config)

    # Create some test terms
    A = Term("mortal", "mortals")
    B = Term("human", "humans")
    C = Term("animal", "animals")

    # Test Barbara (AAA-1)
    # Major premise: All M are P
    # Minor premise: All S are M
    # Conclusion:    All S are P
    assert dataset._is_valid_syllogism(
        (Quantifier.ALL, B, C),  # All B (M) are C (P)  [Major premise]
        (Quantifier.ALL, A, B),  # All A (S) are B (M)  [Minor premise]
        (Quantifier.ALL, A, C),  # All A (S) are C (P)  [Conclusion]
    )

    # Test Celarent (EAE-1)
    # Major premise: No M are P
    # Minor premise: All S are M
    # Conclusion:    No S are P
    assert dataset._is_valid_syllogism(
        (Quantifier.NO, B, C),  # No B (M) are C (P)
        (Quantifier.ALL, A, B),  # All A (S) are B (M)
        (Quantifier.NO, A, C),  # No A (S) are C (P)
    )

    # Test invalid forms
    assert not dataset._is_valid_syllogism(
        (Quantifier.SOME, B, C),  # Some B are C
        (Quantifier.SOME, A, B),  # Some A are B
        (Quantifier.SOME, A, C),  # Some A are C (invalid: two particular premises)
    )

    assert not dataset._is_valid_syllogism(
        (Quantifier.NO, B, C),  # No B are C
        (Quantifier.NO, A, B),  # No A are B
        (Quantifier.NO, A, C),  # No A are C (invalid: two negative premises)
    )

    # Test specific invalid case with two negative premises
    S = Term("student", "students")
    M = Term("human", "humans")
    P = Term("chef", "chefs")
    assert not dataset._is_valid_syllogism(
        (Quantifier.NO, S, M),  # No students are humans
        (Quantifier.NO, M, P),  # No humans are chefs
        (Quantifier.NO, S, P),  # No students are chefs (invalid!)
    )


def test_syllogism_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = SyllogismConfig(size=5, seed=42)
    dataset = SyllogismDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)

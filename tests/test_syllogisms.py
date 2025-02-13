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
        assert "conclusion" in item["metadata"]
        assert "is_valid" in item["metadata"]
        assert "type" in item["metadata"]

        # For traditional syllogisms, check for premise2
        if item["metadata"]["type"] == "syllogism":
            assert "premise2" in item["metadata"]

        # Verify answer format
        assert item["answer"] in ("Yes", "No")

        # Verify question format
        assert "Consider these statements:" in item["question"]
        assert "1." in item["question"]
        if item["metadata"]["type"] == "syllogism":
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
        (Quantifier.ALL, B, C),  # All B (M) are C (P)
        (Quantifier.ALL, A, B),  # All A (S) are B (M)
        (Quantifier.ALL, A, C),  # All A (S) are C (P)
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

    # Test Cesare (EAE-2) — corrected order
    # Major premise: No P are M
    # Minor premise: All S are M
    # Conclusion:    No S are P
    assert dataset._is_valid_syllogism(
        (Quantifier.NO, C, B),  # No C (P) are B (M)  [Major premise]
        (Quantifier.ALL, A, B),  # All A (S) are B (M) [Minor premise]
        (Quantifier.NO, A, C),  # No A (S) are C (P)
    )

    # Test Darii (AII-1)
    # Major premise: All M are P
    # Minor premise: Some S are M
    # Conclusion:    Some S are P
    assert dataset._is_valid_syllogism(
        (Quantifier.ALL, B, C),  # All B (M) are C (P)
        (Quantifier.SOME, A, B),  # Some A (S) are B (M)
        (Quantifier.SOME, A, C),  # Some A (S) are C (P)
    )

    # Test Disamis (IAI-3)
    # Major premise: Some M are P
    # Minor premise: All M are S
    # Conclusion:    Some S are P
    assert dataset._is_valid_syllogism(
        (Quantifier.SOME, B, C),  # Some B (M) are C (P)
        (Quantifier.ALL, B, A),  # All B (M) are A (S)
        (Quantifier.SOME, A, C),  # Some A (S) are C (P)
    )

    # Test Ferio (EIO-1)
    # Major premise: No M are P
    # Minor premise: Some S are M
    # Conclusion:    Some S are not P
    assert dataset._is_valid_syllogism(
        (Quantifier.NO, B, C),  # No B (M) are C (P)
        (Quantifier.SOME, A, B),  # Some A (S) are B (M)
        (Quantifier.SOME_NOT, A, C),  # Some A (S) are not C (P)
    )

    # Test Festino (EIO-2)
    # Major premise: No P are M
    # Minor premise: Some S are M
    # Conclusion:    Some S are not P
    assert dataset._is_valid_syllogism(
        (Quantifier.NO, C, B),  # No C (P) are B (M)
        (Quantifier.SOME, A, B),  # Some A (S) are B (M)
        (Quantifier.SOME_NOT, A, C),  # Some A (S) are not C (P)
    )

    # Test Datisi (AII-3)
    # Major premise: All M are P
    # Minor premise: Some M are S
    # Conclusion:    Some S are P
    assert dataset._is_valid_syllogism(
        (Quantifier.ALL, B, C),  # All B (M) are C (P)
        (Quantifier.SOME, B, A),  # Some B (M) are A (S)
        (Quantifier.SOME, A, C),  # Some A (S) are C (P)
    )

    # Test Bocardo (OAO-3)
    # Major premise: Some M are not P
    # Minor premise: All M are S
    # Conclusion:    Some S are not P
    assert dataset._is_valid_syllogism(
        (Quantifier.SOME_NOT, B, C),  # Some B (M) are not C (P)
        (Quantifier.ALL, B, A),  # All B (M) are A (S)
        (Quantifier.SOME_NOT, A, C),  # Some A (S) are not C (P)
    )

    # Test Baroco (AOO-2)
    # Major premise: All P are M
    # Minor premise: Some S are not M
    # Conclusion:    Some S are not P
    assert dataset._is_valid_syllogism(
        (Quantifier.ALL, C, B),  # All C (P) are B (M)
        (Quantifier.SOME_NOT, A, B),  # Some A (S) are not B (M)
        (Quantifier.SOME_NOT, A, C),  # Some A (S) are not C (P)
    )

    # Test Camestres (AEE-2)
    # Major premise: All P are M
    # Minor premise: No S are M
    # Conclusion:    No S are P
    assert dataset._is_valid_syllogism(
        (Quantifier.ALL, C, B),  # All C (P) are B (M)
        (Quantifier.NO, A, B),  # No A (S) are B (M)
        (Quantifier.NO, A, C),  # No A (S) are C (P)
    )

    # Test Dimaris (IAI-4)
    # Major premise: Some P are M
    # Minor premise: All M are S
    # Conclusion:    Some S are P
    assert dataset._is_valid_syllogism(
        (Quantifier.SOME, C, B),  # Some C (P) are B (M)
        (Quantifier.ALL, B, A),  # All B (M) are A (S)
        (Quantifier.SOME, A, C),  # Some A (S) are C (P)
    )

    # Test Ferison (EIO-3)
    # Major premise: No M are P
    # Minor premise: Some M are S
    # Conclusion:    Some S are not P
    assert dataset._is_valid_syllogism(
        (Quantifier.NO, B, C),  # No B (M) are C (P)
        (Quantifier.SOME, B, A),  # Some B (M) are A (S)
        (Quantifier.SOME_NOT, A, C),  # Some A (S) are not C (P)
    )

    # Test Fresison (EIO-4)
    # Major premise: No P are M
    # Minor premise: Some M are S
    # Conclusion:    Some S are not P
    assert dataset._is_valid_syllogism(
        (Quantifier.NO, C, B),  # No C (P) are B (M)
        (Quantifier.SOME, B, A),  # Some B (M) are A (S)
        (Quantifier.SOME_NOT, A, C),  # Some A (S) are not C (P)
    )

    # Test Camenes (AEE-4)
    # Major premise: All P are M
    # Minor premise: No M are S
    # Conclusion:    No S are P
    assert dataset._is_valid_syllogism(
        (Quantifier.ALL, C, B),  # All C (P) are B (M)
        (Quantifier.NO, B, A),  # No B (M) are A (S)
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

    child = Term("child", "children")
    animal = Term("animal", "animals")
    doctor = Term("doctor", "doctors")

    # Premise 1: Some children are not animals
    # Premise 2: All animals are doctors
    # Conclusion: Some children are not doctors
    # We expect this NOT to be a valid syllogism
    assert not dataset._is_valid_syllogism(
        (Quantifier.SOME_NOT, child, animal),  # Some children are not animals
        (Quantifier.ALL, animal, doctor),  # All animals are doctors
        (Quantifier.SOME_NOT, child, doctor),  # Some children are not doctors
    )


def test_logical_equivalence():
    """Test logical equivalence rules for inversions"""
    config = SyllogismConfig(size=1, seed=42)
    dataset = SyllogismDataset(config)

    # Create test terms
    A = Term("student", "students")
    B = Term("human", "humans")

    # Test direct inversion of NO statements
    assert dataset._check_logical_equivalence(
        (Quantifier.NO, A, B),  # No students are humans
        (Quantifier.NO, B, A),  # No humans are students
    )

    # Test particular inversion of ALL statements
    assert dataset._check_logical_equivalence(
        (Quantifier.ALL, A, B),  # All students are humans
        (Quantifier.SOME, B, A),  # Some humans are students
    )

    # Test direct inversion of SOME statements
    assert dataset._check_logical_equivalence(
        (Quantifier.SOME, A, B),  # Some students are humans
        (Quantifier.SOME, B, A),  # Some humans are students
    )

    # Test invalid inversions
    assert not dataset._check_logical_equivalence(
        (Quantifier.SOME_NOT, A, B),  # Some students are not humans
        (Quantifier.SOME_NOT, B, A),  # Some humans are not students (invalid)
    )

    assert not dataset._check_logical_equivalence(
        (Quantifier.ALL, A, B),  # All students are humans
        (Quantifier.ALL, B, A),  # All humans are students (invalid)
    )


def test_inversion_generation():
    """Test generation of inversion problems"""
    # Force inversion problems by setting probability to 1.0
    config = SyllogismConfig(size=10, seed=42, inversion_probability=1.0)
    dataset = SyllogismDataset(config)

    for item in dataset:
        # Check type is marked as inversion
        assert item["metadata"]["type"] == "inversion"
        # Check both premises and selection
        assert "premise1" in item["metadata"]
        assert "premise2" in item["metadata"]
        assert "selected_premise" in item["metadata"]
        assert item["metadata"]["selected_premise"] in (1, 2)
        # Check format
        assert item["answer"] in ("Yes", "No")
        assert "Consider these statements:" in item["question"]
        assert "1." in item["question"]
        assert "2." in item["question"]  # Inversion questions now show both premises
        assert "Does it logically follow that:" in item["question"]


def test_syllogism_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = SyllogismConfig(size=5, seed=42)
    dataset = SyllogismDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)

import pytest

from reasoning_gym import create_dataset
from reasoning_gym.graphs.family_relationships import (
    FamilyRelationshipsConfig,
    FamilyRelationshipsCurriculum,
    FamilyRelationshipsDataset,
    Relationship,
)


def test_family_relationships_generation():
    dataset = create_dataset("family_relationships", seed=42, size=10)
    assert isinstance(dataset, FamilyRelationshipsDataset)

    for item in dataset:
        # Check required keys exist
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Validate story and question format
        story_and_question = item["question"]
        assert "is married to" in story_and_question
        assert "have" in story_and_question
        assert any(prompt in story_and_question for prompt in ["What is", "How is", "What relation is"])

        # Validate answer is a valid relationship
        assert item["answer"] in [r.value for r in Relationship]

        # Validate metadata
        assert "person1" in item["metadata"]
        assert "person2" in item["metadata"]
        assert "relationship" in item["metadata"]
        assert "family_size" in item["metadata"]
        assert item["metadata"]["family_size"] >= 4  # Minimum family size


def test_family_relationships_config():
    # Test invalid config raises assertion
    with pytest.raises(AssertionError):
        dataset = create_dataset("family_relationships", min_family_size=2)

    with pytest.raises(AssertionError):
        dataset = create_dataset("family_relationships", max_family_size=3, min_family_size=4)

    with pytest.raises(AssertionError):
        dataset = create_dataset("family_relationships", male_names=[])

    with pytest.raises(AssertionError):
        dataset = create_dataset("family_relationships", female_names=[])


def test_deterministic_generation():
    dataset1 = create_dataset("family_relationships", seed=42, size=5)
    dataset2 = create_dataset("family_relationships", seed=42, size=5)

    for i in range(5):
        assert dataset1[i]["question"] == dataset2[i]["question"]
        assert dataset1[i]["answer"] == dataset2[i]["answer"]


def test_relationship_consistency():
    dataset = create_dataset("family_relationships", seed=42, size=10)

    for item in dataset:
        # Check that relationship matches the gender
        relationship = item["metadata"]["relationship"]
        if relationship in [
            Relationship.MOTHER.value,
            Relationship.GRANDMOTHER.value,
            Relationship.WIFE.value,
            Relationship.SISTER.value,
            Relationship.DAUGHTER.value,
            Relationship.AUNT.value,
            Relationship.NIECE.value,
            Relationship.MOTHER_IN_LAW.value,
        ]:
            assert "married to" in item["question"] or "child" in item["question"]
        elif relationship in [
            Relationship.FATHER.value,
            Relationship.GRANDFATHER.value,
            Relationship.HUSBAND.value,
            Relationship.BROTHER.value,
            Relationship.SON.value,
            Relationship.UNCLE.value,
            Relationship.NEPHEW.value,
            Relationship.FATHER_IN_LAW.value,
        ]:
            assert "married to" in item["question"] or "child" in item["question"]


def test_family_relationships_curriculum():
    """Test the family relationships curriculum functionality"""
    curriculum = FamilyRelationshipsCurriculum()

    base_value = {"size": 50, "seed": 42}

    # Test default configuration
    base_cfg: FamilyRelationshipsConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 42
    assert base_cfg.size == 50
    assert base_cfg.min_family_size == 3 and base_cfg.max_family_size == 3  # Default level 0

    # Test incrementing family_size attribute
    curriculum.increment_attr_level("family_size")
    first_level_cfg = curriculum.generate_configuration(base_value)
    assert first_level_cfg.min_family_size == 3 and first_level_cfg.max_family_size == 4  # Level 1

    # Test incrementing family_size attribute again
    curriculum.increment_attr_level("family_size")
    second_level_cfg = curriculum.generate_configuration(base_value)
    assert second_level_cfg.min_family_size == 3 and second_level_cfg.max_family_size == 5  # Level 2

    # Test decrementing family_size attribute
    curriculum.decrement_attr_level("family_size")
    back_to_first_cfg = curriculum.generate_configuration(base_value)
    assert back_to_first_cfg.min_family_size == 3 and back_to_first_cfg.max_family_size == 4  # Back to level 1

    # Test global level setting
    curriculum.set_global_level(5)  # Set to level 5
    level_five_cfg = curriculum.generate_configuration(base_value)
    assert level_five_cfg.min_family_size == 3 and level_five_cfg.max_family_size == 8  # Level 5

    # Test increment global level
    curriculum.increment_global_level()  # Should go to level 6
    level_six_cfg = curriculum.generate_configuration(base_value)
    assert level_six_cfg.min_family_size == 3 and level_six_cfg.max_family_size == 9  # Level 6

    # Test decrement global level
    curriculum.decrement_global_level()  # Should go back to level 5
    back_to_five_cfg = curriculum.generate_configuration(base_value)
    assert back_to_five_cfg.min_family_size == 3 and back_to_five_cfg.max_family_size == 8  # Back to level 5

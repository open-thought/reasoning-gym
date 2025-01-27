import pytest

from reasoning_gym import create_dataset
from reasoning_gym.graphs.family_relationships import FamilyRelationshipsDataset, Relationship


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

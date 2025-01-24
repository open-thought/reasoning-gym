from reasoning_gym.graphs.family_relationships import (
    family_relationships_dataset,
    Gender,
    Relationship,
)


def test_family_relationships_generation():
    dataset = family_relationships_dataset(seed=42, size=10)
    
    for item in dataset:
        # Check required keys exist
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        
        # Validate story and question format
        story_and_question = item["question"]
        assert "is married to" in story_and_question
        assert "have" in story_and_question
        assert any(
            prompt in story_and_question 
            for prompt in [
                "What is",
                "How is",
                "What relation is"
            ]
        )
        
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
        dataset = family_relationships_dataset(min_family_size=2)
    
    with pytest.raises(AssertionError):
        dataset = family_relationships_dataset(max_family_size=3, min_family_size=4)
    
    with pytest.raises(AssertionError):
        dataset = family_relationships_dataset(male_names=[])
    
    with pytest.raises(AssertionError):
        dataset = family_relationships_dataset(female_names=[])


def test_deterministic_generation():
    dataset1 = family_relationships_dataset(seed=42, size=5)
    dataset2 = family_relationships_dataset(seed=42, size=5)
    
    for i in range(5):
        assert dataset1[i]["question"] == dataset2[i]["question"]
        assert dataset1[i]["answer"] == dataset2[i]["answer"]


def test_relationship_consistency():
    dataset = family_relationships_dataset(seed=42, size=10)
    
    for item in dataset:
        # Check that relationship matches the gender
        relationship = item["metadata"]["relationship"]
        if relationship in [Relationship.MOTHER.value, Relationship.GRANDMOTHER.value, 
                          Relationship.WIFE.value, Relationship.SISTER.value, 
                          Relationship.DAUGHTER.value]:
            assert "married to" in item["question"] or "child" in item["question"]
        elif relationship in [Relationship.FATHER.value, Relationship.GRANDFATHER.value,
                            Relationship.HUSBAND.value, Relationship.BROTHER.value,
                            Relationship.SON.value]:
            assert "married to" in item["question"] or "child" in item["question"]

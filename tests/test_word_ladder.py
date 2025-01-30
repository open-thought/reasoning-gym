import pytest

from reasoning_gym.algorithmic.word_ladder import WordLadderConfig, WordLadderDataset


def test_word_ladder_config_validation():
    """Test that invalid configs raise appropriate errors"""
    # Test min_word_length validation
    with pytest.raises(AssertionError):
        config = WordLadderConfig(min_word_length=2)
        config.validate()

    # Test max_word_length validation
    with pytest.raises(AssertionError):
        config = WordLadderConfig(max_word_length=6)
        config.validate()

    # Test word length relationship
    with pytest.raises(AssertionError):
        config = WordLadderConfig(min_word_length=5, max_word_length=3)
        config.validate()

    # Test min_chain_length validation
    with pytest.raises(AssertionError):
        config = WordLadderConfig(min_chain_length=2)
        config.validate()

    # Test chain length relationship
    with pytest.raises(AssertionError):
        config = WordLadderConfig(min_chain_length=5, max_chain_length=3)
        config.validate()


def test_word_ladder_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = WordLadderConfig(seed=42, size=10)
    dataset1 = WordLadderDataset(config)
    dataset2 = WordLadderDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_word_ladder_dataset_items():
    """Test basic properties of generated items"""
    config = WordLadderConfig(
        min_word_length=3, max_word_length=5, min_chain_length=3, max_chain_length=5, size=10, seed=42
    )
    dataset = WordLadderDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        metadata = item["metadata"]
        assert "start_word" in metadata
        assert "end_word" in metadata
        assert "word_length" in metadata
        assert "chain_length" in metadata

        # Verify word length constraints
        word_length = metadata["word_length"]
        assert config.min_word_length <= word_length <= config.max_word_length
        assert len(metadata["start_word"]) == word_length
        assert len(metadata["end_word"]) == word_length

        # Verify solution chain from answer
        solution_chain = item["answer"].split(",")

        # Handle chain length validation based on whether it's shortest path (-1) or specified length
        if metadata["chain_length"] == -1:
            # For shortest path, just ensure it's a valid path (we can't predict exact length)
            assert len(solution_chain) >= 2  # Must have at least start and end words
        else:
            # For specified length, ensure it matches config constraints
            assert config.min_chain_length <= len(solution_chain) <= config.max_chain_length
            assert len(solution_chain) == metadata["chain_length"]

        assert solution_chain[0] == metadata["start_word"]
        assert solution_chain[-1] == metadata["end_word"]
        assert all(len(word) == word_length for word in solution_chain)

        # Verify each step differs by only one letter
        for j in range(len(solution_chain) - 1):
            differences = sum(1 for a, b in zip(solution_chain[j], solution_chain[j + 1]) if a != b)
            assert differences == 1


def test_word_ladder_differs_by_one():
    """Test the _differs_by_one helper method"""
    config = WordLadderConfig()
    dataset = WordLadderDataset(config)

    # Test words that differ by one letter
    assert dataset._differs_by_one("CAT", "BAT")
    assert dataset._differs_by_one("DOG", "LOG")
    assert dataset._differs_by_one("WORD", "WARD")

    # Test words that differ by more than one letter
    assert not dataset._differs_by_one("CAT", "DOG")
    assert not dataset._differs_by_one("WORD", "WAND")

    # Test words of different lengths
    assert not dataset._differs_by_one("CAT", "CATS")
    assert not dataset._differs_by_one("DOG", "DO")

    # Test identical words
    assert not dataset._differs_by_one("CAT", "CAT")


def test_word_ladder_find_path():
    """Test the _find_path helper method"""
    config = WordLadderConfig()
    dataset = WordLadderDataset(config)

    # Create a small test word set
    word_set = {"CAT", "BAT", "BAR", "CAR"}

    # Test finding valid paths
    path1 = dataset._find_path("CAT", "BAR", word_set)
    assert path1 is not None
    assert path1[0] == "CAT"
    assert path1[-1] == "BAR"
    assert all(word in word_set for word in path1)

    # Test when no path exists
    word_set = {"CAT", "DOG"}
    path2 = dataset._find_path("CAT", "DOG", word_set)
    assert path2 is None

    # Test path to same word
    path3 = dataset._find_path("CAT", "CAT", word_set)
    assert path3 == ["CAT"]


if __name__ == "__main__":
    pytest.main([__file__])

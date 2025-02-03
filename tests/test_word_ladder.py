import time
from random import Random

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

    # Test new chain length validation rules
    with pytest.raises(AssertionError):
        # When min_chain_length is -1, max_chain_length must be -1 or >= 3
        config = WordLadderConfig(min_chain_length=-1, max_chain_length=2)
        config.validate()

    with pytest.raises(AssertionError):
        # max_chain_length can't be -1 unless min_chain_length is also -1
        config = WordLadderConfig(min_chain_length=3, max_chain_length=-1)
        config.validate()

    with pytest.raises(AssertionError):
        # min_chain_length must be 3 or -1
        config = WordLadderConfig(min_chain_length=2)
        config.validate()

    with pytest.raises(AssertionError):
        # max_chain_length must be >= min_chain_length
        config = WordLadderConfig(min_chain_length=5, max_chain_length=3)
        config.validate()

    # Test dataset size validation
    with pytest.raises(ValueError):
        config = WordLadderConfig(min_word_length=3, max_word_length=3, size=1000000)
        config.validate()


def test_word_ladder_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = WordLadderConfig(seed=42, size=10)
    dataset1 = WordLadderDataset(config)
    dataset2 = WordLadderDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_word_ladder_dataset_unique_pairs():
    """Test that generated word pairs are unique"""
    config = WordLadderConfig(size=50, seed=42)
    dataset = WordLadderDataset(config)

    # Track all generated pairs
    seen_pairs = set()
    for i in range(len(dataset)):
        item = dataset[i]
        pair = (
            min(item["metadata"]["start_word"], item["metadata"]["end_word"]),
            max(item["metadata"]["start_word"], item["metadata"]["end_word"]),
        )
        assert pair not in seen_pairs, f"Duplicate pair found: {pair}"
        seen_pairs.add(pair)


def test_word_ladder_dataset_items():
    """Test basic properties of generated items"""
    # Test with specific chain length constraints
    config1 = WordLadderConfig(
        min_word_length=3, max_word_length=5, min_chain_length=3, max_chain_length=5, size=10, seed=42
    )
    dataset1 = WordLadderDataset(config1)

    # Test with shortest path mode
    config2 = WordLadderConfig(
        min_word_length=3, max_word_length=5, min_chain_length=-1, max_chain_length=-1, size=10, seed=42
    )
    dataset2 = WordLadderDataset(config2)

    for dataset in [dataset1, dataset2]:
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
            assert dataset.config.min_word_length <= word_length <= dataset.config.max_word_length
            assert len(metadata["start_word"]) == word_length
            assert len(metadata["end_word"]) == word_length

            # Verify solution chain from answer
            solution_chain = item["answer"].split(",")

            # Verify chain length based on config
            if dataset.config.min_chain_length == -1:
                assert len(solution_chain) >= 3
            else:
                assert dataset.config.min_chain_length <= len(solution_chain) <= dataset.config.max_chain_length

            assert solution_chain[0] == metadata["start_word"]
            assert solution_chain[-1] == metadata["end_word"]

            # Verify each step differs by only one letter using new neighbor logic
            for j in range(len(solution_chain) - 1):
                current = solution_chain[j]
                next_word = solution_chain[j + 1]
                word_set = dataset.word_sets[len(current)]
                assert next_word in dataset._get_neighbors(current, word_set)


def test_word_ladder_get_neighbors():
    """Test the _get_neighbors helper method"""
    config = WordLadderConfig()
    dataset = WordLadderDataset(config)

    # Create a test word set
    word_set = {"CAT", "BAT", "RAT", "CAR", "DOG"}

    # Test finding neighbors
    neighbors = dataset._get_neighbors("CAT", word_set)
    assert neighbors == {"BAT", "RAT", "CAR"}

    # Test with no neighbors
    neighbors = dataset._get_neighbors("DOG", word_set)
    assert len(neighbors) == 0

    # Test with empty word set
    neighbors = dataset._get_neighbors("CAT", set())
    assert len(neighbors) == 0


def test_word_ladder_path_finding():
    """Test the basic path finding capabilities"""
    config = WordLadderConfig(
        min_word_length=4,
        max_word_length=4,
        min_chain_length=-1,  # Shortest path mode
        max_chain_length=-1,
        size=10,
        seed=42,
    )
    dataset = WordLadderDataset(config)

    # Test finding path between known words
    word_set = dataset.word_sets[4]
    path = dataset._find_path("WORD", "FIND", word_set)

    # Verify path properties
    assert path is not None
    assert path[0] == "WORD"
    assert path[-1] == "FIND"
    assert len(path) >= 3

    # Verify each step differs by only one letter
    for i in range(len(path) - 1):
        current = path[i]
        next_word = path[i + 1]
        assert next_word in dataset._get_neighbors(current, word_set)


def test_word_ladder_csv_loading():
    """Test word loading from CSV"""
    config = WordLadderConfig(min_word_length=3, max_word_length=5)
    dataset = WordLadderDataset(config)

    # Verify word sets for each length
    for length in range(3, 6):
        assert length in dataset.word_sets
        words = dataset.word_sets[length]
        assert len(words) > 0
        # Verify all words are correct length and uppercase
        for word in words:
            assert len(word) == length
            assert word.isupper()
            assert word.isalpha()

    # Test invalid length range
    with pytest.raises(AssertionError):
        bad_config = WordLadderConfig(min_word_length=2, max_word_length=7)
        WordLadderDataset(bad_config)


def test_word_ladder_pair_generation():
    """Test word pair generation logic"""
    config = WordLadderConfig(min_word_length=4, max_word_length=4, size=10, seed=42)
    dataset = WordLadderDataset(config)

    # Test pair generation
    rng = Random(42)
    start, end, path = dataset._generate_word_pair(rng, 4)

    # Verify path properties
    assert start == path[0]
    assert end == path[-1]
    assert len(path) >= 3

    # Verify path is valid (each step differs by one letter)
    for i in range(len(path) - 1):
        current = path[i]
        next_word = path[i + 1]
        assert next_word in dataset._get_neighbors(current, dataset.word_sets[4])


def test_word_graph_caching():
    """Test word graph caching functionality"""
    config = WordLadderConfig(seed=42)
    dataset = WordLadderDataset(config)

    # Verify initial graphs are precomputed
    assert len(dataset.word_graphs) > 0

    # Get initial graph for length 4
    graph_4 = dataset.word_graphs[4]

    # Verify cached graph is returned
    cached_graph = dataset._build_word_graph(4)
    assert cached_graph is dataset.word_graphs[4]

    # Verify graph structure
    for word, neighbors in graph_4.items():
        assert len(word) == 4
        for neighbor in neighbors:
            assert len(neighbor) == 4
            # Verify bidirectional connections
            assert word in graph_4[neighbor]


def test_word_ladder_path_validation():
    """Test path length validation logic"""
    config = WordLadderConfig(min_chain_length=4, max_chain_length=6)

    # Test specific length mode
    assert config.is_valid_path_length(4)  # Min length
    assert config.is_valid_path_length(5)  # Middle length
    assert config.is_valid_path_length(6)  # Max length
    assert not config.is_valid_path_length(3)  # Too short
    assert not config.is_valid_path_length(7)  # Too long

    # Test shortest path mode
    config_shortest = WordLadderConfig(min_chain_length=-1, max_chain_length=-1)
    assert config_shortest.is_valid_path_length(3)
    assert config_shortest.is_valid_path_length(4)
    assert config_shortest.is_valid_path_length(10)
    assert not config_shortest.is_valid_path_length(2)

    # Test mixed mode (shortest with max limit)
    config_mixed = WordLadderConfig(min_chain_length=-1, max_chain_length=5)
    assert config_mixed.is_valid_path_length(3)
    assert config_mixed.is_valid_path_length(4)
    assert config_mixed.is_valid_path_length(5)
    assert not config_mixed.is_valid_path_length(6)


def test_word_ladder_solution_optimality():
    """Test that generated solutions are optimal when min_chain_length=-1"""
    config = WordLadderConfig(
        min_word_length=4, max_word_length=4, min_chain_length=-1, max_chain_length=-1, size=20, seed=42
    )
    dataset = WordLadderDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        solution_chain = item["answer"].split(",")
        start_word = item["metadata"]["start_word"]
        end_word = item["metadata"]["end_word"]

        # Verify this is the shortest possible path
        word_set = dataset.word_sets[len(start_word)]

        # Build graph and use BFS to find shortest path
        from collections import deque

        queue = deque([(start_word, [start_word])])
        visited = {start_word}
        shortest_path = None

        while queue and not shortest_path:
            current_word, path = queue.popleft()
            if current_word == end_word:
                shortest_path = path
                break

            for neighbor in dataset._get_neighbors(current_word, word_set):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        assert shortest_path is not None, f"No path found between {start_word} and {end_word}"
        assert len(solution_chain) == len(
            shortest_path
        ), f"Solution {solution_chain} is not optimal. Shortest path: {shortest_path}"


def test_word_ladder_performance():
    """Test performance with large datasets"""
    config = WordLadderConfig(size=100)
    start_time = time.time()
    dataset = WordLadderDataset(config)
    init_time = time.time() - start_time

    # Test item generation time
    start_time = time.time()
    for i in range(len(dataset)):
        _ = dataset[i]
    access_time = time.time() - start_time

    # These thresholds should be adjusted based on requirements
    assert init_time < 2.0, f"Initialization took too long: {init_time:.2f}s"
    assert access_time < 1.0, f"Data access took too long: {access_time:.2f}s"


def test_word_ladder_edge_cases():
    """Test edge cases and corner conditions"""
    # Test with minimum possible size
    config = WordLadderConfig(size=1)
    dataset = WordLadderDataset(config)
    assert len(dataset) == 1

    # Test with same start/end word length but maximum distance
    config = WordLadderConfig(min_word_length=4, max_word_length=4, min_chain_length=-1, max_chain_length=-1, size=10)
    dataset = WordLadderDataset(config)

    # Find the pair with longest solution
    max_length = 0
    for i in range(len(dataset)):
        item = dataset[i]
        chain_length = len(item["answer"].split(","))
        max_length = max(max_length, chain_length)

    assert max_length > 3, "No challenging word pairs generated"


if __name__ == "__main__":
    pytest.main([__file__])

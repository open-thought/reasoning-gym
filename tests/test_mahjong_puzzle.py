"""Tests for Mahjong Puzzle questions generation"""

import string

import pytest

from reasoning_gym.games.mahjong import MahjongPuzzleConfig, MahjongPuzzleDataset


def test_mahjong_puzzle_config_validation():
    """Test that invalid configs raise appropriate errors"""

    with pytest.raises(AssertionError):
        config = MahjongPuzzleConfig(min_num_rounds=-1)  # Negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = MahjongPuzzleConfig(min_num_rounds=0)  # Zero not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = MahjongPuzzleConfig(min_num_rounds=3, max_num_rounds=2)  # Min > Max
        config.validate()


def test_mahjong_puzzle_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = MahjongPuzzleConfig(seed=42, size=10)
    dataset1 = MahjongPuzzleDataset(config)
    dataset2 = MahjongPuzzleDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_mahjong_puzzle_dataset_items():
    """Test basic properties of generated items"""
    config = MahjongPuzzleConfig(min_num_rounds=3, max_num_rounds=5, size=10, seed=42)
    dataset = MahjongPuzzleDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "rounds" in item["metadata"]
        assert "solution" in item["metadata"]

        rounds = item["metadata"]["rounds"]
        solution = item["metadata"]["solution"]

        # Verify values
        assert solution in ["Peng", "Chi", "Pass"]
        assert 3 <= len(rounds) <= 5
        assert all(isinstance(r, dict) for r in rounds)
        assert all("add" in r for r in rounds)
        assert all("remove" in r for r in rounds)
        assert all(len(r["cards"]) == 13 for r in rounds)
        assert all(r["result"] in ["Peng", "Chi", "Pass"] for r in rounds)


def test_mahjong_puzzle_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = MahjongPuzzleConfig(size=5, seed=42)
    dataset = MahjongPuzzleDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_mahjong_puzzle_answer():
    """Test the _get_answer method"""
    config = MahjongPuzzleConfig(seed=42)
    dataset = MahjongPuzzleDataset(config)

    # Peng
    cards = "ABBCCDDEEFFGH"
    assert dataset._check_peng(cards, new_card="B") == True  # B, B, B
    assert dataset._check_peng(cards, new_card="A") == False

    # Chi
    cards = "ABDGIKMOQSUWY"
    assert dataset._check_chi(cards, new_card="C") == True  # A, B, C
    assert dataset._check_chi(cards, new_card="A") == False

    # Pass
    cards = "ACEGIKMOQSUWY"
    for c in string.ascii_lowercase:
        assert dataset._check_peng(cards, new_card=c) == False
        assert dataset._check_chi(cards, new_card=c) == False

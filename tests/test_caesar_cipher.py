"""Tests for Caesar cipher task generation"""

import pytest

from reasoning_gym.algorithmic.caesar_cipher import CaesarCipherConfig, CaesarCipherDataset


def test_caesar_cipher_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = CaesarCipherConfig(min_words=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = CaesarCipherConfig(min_words=10, max_words=5)
        config.validate()

    with pytest.raises(AssertionError):
        config = CaesarCipherConfig(min_rotation=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = CaesarCipherConfig(max_rotation=26)
        config.validate()


def test_caesar_cipher_deterministic():
    """Test that dataset generates same items with same seed"""
    config = CaesarCipherConfig(seed=42, size=10)
    dataset1 = CaesarCipherDataset(config)
    dataset2 = CaesarCipherDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_caesar_cipher_encryption():
    """Test the Caesar cipher encryption logic"""
    config = CaesarCipherConfig(size=1, seed=42)
    dataset = CaesarCipherDataset(config)

    # Test with known rotation
    text = "HELLO"
    encrypted = dataset._caesar_encrypt(text, 1)
    assert encrypted == "IFMMP"  # Each letter shifted by 1

    # Test wrapping around Z
    encrypted = dataset._caesar_encrypt("XYZ", 2)
    assert encrypted == "ZAB"

    # Test preserving spaces
    encrypted = dataset._caesar_encrypt("HELLO WORLD", 1)
    assert encrypted == "IFMMP XPSME"


def test_caesar_cipher_dataset_items():
    """Test basic properties of generated items"""
    config = CaesarCipherConfig(min_words=3, max_words=5, min_rotation=1, max_rotation=3, size=10, seed=42)
    dataset = CaesarCipherDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]

        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "rotation" in item["metadata"]
        assert "cipher_text" in item["metadata"]
        assert "clear_text" in item["metadata"]

        # Verify rotation constraints
        rotation = item["metadata"]["rotation"]
        assert config.min_rotation <= rotation <= config.max_rotation

        # Verify text properties
        clear_text = item["metadata"]["clear_text"]
        words = clear_text.split()
        assert config.min_words <= len(words) <= config.max_words
        assert all(word.isupper() and word.isalpha() for word in words)

        # Verify encryption
        cipher_text = item["metadata"]["cipher_text"]
        decrypted = dataset._caesar_encrypt(cipher_text, -rotation)  # Decrypt by negative rotation
        assert decrypted == clear_text


def test_caesar_cipher_iteration():
    """Test that iteration respects dataset size"""
    config = CaesarCipherConfig(size=5, seed=42)
    dataset = CaesarCipherDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)

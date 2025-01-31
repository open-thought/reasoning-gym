"""Tests for base conversion task generation"""

import pytest

from reasoning_gym.algorithmic.base_conversion import BaseConversionConfig, BaseConversionDataset


def test_base_conversion_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = BaseConversionConfig(min_base=1)  # Too small
        config.validate()

    with pytest.raises(AssertionError):
        config = BaseConversionConfig(min_base=37)  # Too large
        config.validate()

    with pytest.raises(AssertionError):
        config = BaseConversionConfig(min_base=10, max_base=5)  # max < min
        config.validate()

    with pytest.raises(AssertionError):
        config = BaseConversionConfig(min_value=-1)  # Negative not allowed
        config.validate()


def test_base_conversion_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = BaseConversionConfig(seed=42, size=10)
    dataset1 = BaseConversionDataset(config)
    dataset2 = BaseConversionDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_base_conversion_dataset_items():
    """Test basic properties of generated items"""
    config = BaseConversionConfig(min_base=2, max_base=16, min_value=0, max_value=1000, size=10, seed=42)
    dataset = BaseConversionDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "decimal_value" in item["metadata"]
        assert "source_base" in item["metadata"]
        assert "target_base" in item["metadata"]
        assert "source_repr" in item["metadata"]
        assert "target_repr" in item["metadata"]

        # Verify value range
        assert config.min_value <= item["metadata"]["decimal_value"] <= config.max_value

        # Verify base range
        assert config.min_base <= item["metadata"]["source_base"] <= config.max_base
        assert config.min_base <= item["metadata"]["target_base"] <= config.max_base
        assert item["metadata"]["source_base"] != item["metadata"]["target_base"]

        # Verify conversion correctness
        decimal_value = item["metadata"]["decimal_value"]
        target_base = item["metadata"]["target_base"]

        # Use same conversion logic as implementation
        if target_base == 16:
            expected = format(decimal_value, "x")
        elif target_base == 2:
            expected = format(decimal_value, "b")
        else:
            # Manual conversion for other bases
            n = decimal_value
            digits = []
            while n:
                digits.append(int(n % target_base))
                n //= target_base
            expected = "".join(str(d) if d < 10 else chr(ord("a") + d - 10) for d in reversed(digits) or [0])
        assert item["answer"] == expected


def test_base_conversion_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = BaseConversionConfig(size=5, seed=42)
    dataset = BaseConversionDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_base_conversion_validity():
    """Test that generated numbers are valid for their bases"""
    config = BaseConversionConfig(min_base=2, max_base=36, min_value=0, max_value=1000, size=100, seed=42)
    dataset = BaseConversionDataset(config)

    def is_valid_for_base(num_str: str, base: int) -> bool:
        valid_chars = "0123456789abcdefghijklmnopqrstuvwxyz"[:base]
        return all(c in valid_chars for c in num_str.lower())

    for i in range(len(dataset)):
        item = dataset[i]
        assert is_valid_for_base(
            item["metadata"]["source_repr"], item["metadata"]["source_base"]
        ), f"Invalid source number {item['metadata']['source_repr']} for base {item['metadata']['source_base']}"
        assert is_valid_for_base(
            item["metadata"]["target_repr"], item["metadata"]["target_base"]
        ), f"Invalid target number {item['metadata']['target_repr']} for base {item['metadata']['target_base']}"


def test_base_conversion_special_bases():
    """Test conversion between special bases (binary, hex)"""
    config = BaseConversionConfig(
        min_base=2,
        max_base=16,
        min_value=0,
        max_value=255,  # Use small range for predictable results
        size=100,
        seed=42,
    )
    dataset = BaseConversionDataset(config)

    binary_found = False
    hex_found = False

    for i in range(len(dataset)):
        item = dataset[i]
        if item["metadata"]["target_base"] == 2:
            binary_found = True
            # Verify binary format
            assert all(c in "01" for c in item["answer"])
        elif item["metadata"]["target_base"] == 16:
            hex_found = True
            # Verify hex format
            assert all(c in "0123456789abcdef" for c in item["answer"])

    assert binary_found, "No binary conversion tasks generated"
    assert hex_found, "No hexadecimal conversion tasks generated"


def test_base_conversion_formatting():
    """Test number formatting in different bases"""
    config = BaseConversionConfig(
        min_base=11,  # Force bases that use letters
        max_base=36,
        min_value=10,  # Ensure multi-digit numbers
        max_value=1000,
        size=10,
        seed=42,
    )
    dataset = BaseConversionDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Verify lowercase letters are used
        assert item["answer"] == item["answer"].lower()
        # Verify no whitespace in answer
        assert item["answer"].strip() == item["answer"]
        # Verify hint is included for bases > 10
        assert "use lowercase letters" in item["question"]

from random import Random

import pytest

from reasoning_gym.arithmetic.gsm_symbolic import GSMSymbolicDataset, GSMSymbolicDatasetConfig


def test_gsm_symbolic_config_validation():
    """Test that config validation works"""
    config = GSMSymbolicDatasetConfig(size=-1)
    with pytest.raises(AssertionError):
        config.validate()


def test_gsm_symbolic_deterministic():
    """Test that dataset generates same items with same seed"""
    config = GSMSymbolicDatasetConfig(seed=42, size=10)
    dataset1 = GSMSymbolicDataset(config)
    dataset2 = GSMSymbolicDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_gsm_symbolic_items():
    """Test basic properties of generated items"""
    config = GSMSymbolicDatasetConfig(size=100, seed=42)
    dataset = GSMSymbolicDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert isinstance(item["question"], str)
        assert isinstance(item["answer"], str)


def test_gsm_symbolic_iteration():
    """Test that iteration respects dataset size"""
    config = GSMSymbolicDatasetConfig(size=5, seed=42)  # Small size for testing
    dataset = GSMSymbolicDataset(config)

    # Test manual iteration
    items = []
    for item in dataset:
        items.append(item)
    assert len(items) == config.size, "Iterator should yield exactly size items"

    # Test list conversion
    items = list(dataset)
    assert len(items) == config.size, "Iterator should yield exactly size items"

    # Test multiple iterations
    first_items = list(dataset)
    second_items = list(dataset)
    assert first_items == second_items, "Multiple iterations should yield same items"


def test_gsm_symbolic_generators():
    """Test generator loading and access"""
    config = GSMSymbolicDatasetConfig()
    dataset = GSMSymbolicDataset(config)

    # Test lazy loading
    assert dataset._generators is None
    _ = dataset.generators  # Access to trigger loading
    assert dataset._generators is not None

    # Test generator mapping
    assert isinstance(dataset.generators, dict)
    assert len(dataset.generators) > 0
    i = 0
    rng = Random(18)
    for key in sorted(dataset.generators.keys()):
        generator = dataset.generators[key]
        assert callable(generator)

        print(i, key)
        answer_set = set()
        question_set = set()
        for j in range(10):
            x = generator(rng, difficulty=1.0)
            question_set.add(x["question"])
            answer_set.add(x["answer"])
            # if j == 123:
            #     print(f"[{j}] q: {x['question']}")
            #     print(f"a: {x['answer']}")
            #     print()
        print(f"ok: q={len(question_set)}, a={len(answer_set)}")

        i += 1


def test_gsm_symbolic_score_answer():
    """Test score answer function"""
    config = GSMSymbolicDatasetConfig(size=100, seed=42)
    dataset = GSMSymbolicDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        score = dataset.score_answer(item["answer"], item)
        assert score == 1.0


def test_gsm_symbolic_score_answer_thousands_separator():
    """Answers written with thousands separators get full credit"""
    config = GSMSymbolicDatasetConfig(size=100, seed=42)
    dataset = GSMSymbolicDataset(config)

    checked = 0
    for item in dataset:
        value = float(item["answer"])
        if value >= 1000 and value.is_integer():
            assert dataset.score_answer(f"{int(value):,}", item) == 1.0
            assert dataset.score_answer(f"{int(value) + 1:,}", item) == 0.01
            checked += 1
    assert checked > 0

    item = dataset[0]
    assert dataset.score_answer("1,234,567.5", {**item, "answer": "1234567.5"}) == 1.0

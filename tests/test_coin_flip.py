from fractions import Fraction

import pytest

from reasoning_gym.probability import CoinFlipConfig, CoinFlipCurriculum, CoinFlipDataset


def test_coin_flip_config_validation():
    """Test that invalid configs raise errors"""
    with pytest.raises(AssertionError):
        config = CoinFlipConfig(size=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = CoinFlipConfig(min_trials=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = CoinFlipConfig(min_trials=5, max_trials=3)
        config.validate()

    with pytest.raises(AssertionError):
        config = CoinFlipConfig(allow_exact=False, allow_at_least=False)
        config.validate()


def test_coin_flip_deterministic():
    """Dataset generates same items with same seed"""
    config = CoinFlipConfig(size=10, seed=42)
    dataset1 = CoinFlipDataset(config)
    dataset2 = CoinFlipDataset(config)
    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_coin_flip_items():
    """Test basic properties of generated items"""
    config = CoinFlipConfig(min_trials=3, max_trials=6, size=7, seed=42)
    dataset = CoinFlipDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert 0.0 <= float(item["answer"]) <= 1.0
        assert "metadata" in item

        metadata = item["metadata"]
        assert "num_trials" in metadata
        assert "k_heads" in metadata
        assert "problem_type" in metadata
        assert metadata["problem_type"] in ["exact", "at_least"]

        rational = metadata["rational"]
        assert rational["denominator"] == 2 ** metadata["num_trials"]
        assert rational["numerator"] > 0


def test_coin_flip_score_answer():
    """Test full and partial reward behavior"""
    config = CoinFlipConfig(size=200, seed=42)
    dataset = CoinFlipDataset(config)

    for i in range(len(dataset)):
        entry = dataset[i]
        answer = entry["answer"]

        # Exact answer -> full reward
        reward = dataset.score_answer(answer, entry)
        assert reward == 1.0

        # Slightly wrong answer -> partial reward
        if float(answer) + 0.01 <= 1.0:
            slightly_wrong = str(float(answer) + 0.01)
        else:
            slightly_wrong = str(float(answer) - 0.01)
        reward_partial = dataset.score_answer(slightly_wrong, entry)
        assert 0.0 <= reward_partial <= 1.0


def test_coin_flip_curriculum():
    """Test curriculum generates valid configurations and increments attributes"""

    curriculum = CoinFlipCurriculum()
    base_value = {"size": 100, "seed": 32}

    cfg = curriculum.generate_configuration(base_value)

    assert isinstance(cfg, CoinFlipConfig)
    assert cfg.size == 100
    assert cfg.seed == 32
    assert cfg.min_trials == 3
    assert cfg.max_trials == 3

    # Increment attribute level for num_trials
    curriculum.increment_attr_level("num_trials")
    cfg_inc = curriculum.generate_configuration(base_value)
    assert cfg_inc.min_trials == 3
    assert cfg_inc.max_trials == 4

    # Decrement attribute level
    curriculum.decrement_attr_level("num_trials")
    cfg_dec = curriculum.generate_configuration(base_value)
    assert cfg_dec.min_trials == 3
    assert cfg_dec.max_trials == 3

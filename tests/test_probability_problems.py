from fractions import Fraction

import pytest

from reasoning_gym.probability.probability_problems import (
    TASK_TYPES,
    ProbabilityProblemsConfig,
    ProbabilityProblemsCurriculum,
    ProbabilityProblemsDataset,
)


def test_config_validation():
    with pytest.raises(AssertionError):
        config = ProbabilityProblemsConfig(min_n=1)
        config.validate()

    with pytest.raises(AssertionError):
        config = ProbabilityProblemsConfig(min_n=10, max_n=5)
        config.validate()

    with pytest.raises(AssertionError):
        config = ProbabilityProblemsConfig(size=0)
        config.validate()


def test_deterministic():
    config = ProbabilityProblemsConfig(seed=42, size=10)
    ds1 = ProbabilityProblemsDataset(config)
    ds2 = ProbabilityProblemsDataset(config)
    for i in range(len(ds1)):
        assert ds1[i] == ds2[i]


def test_item_structure():
    config = ProbabilityProblemsConfig(seed=42, size=50)
    ds = ProbabilityProblemsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert item["metadata"]["source_dataset"] == "probability_problems"


def test_answer_is_valid_fraction():
    config = ProbabilityProblemsConfig(seed=42, size=100)
    ds = ProbabilityProblemsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        frac = Fraction(item["answer"])
        assert frac.denominator > 0


def test_score_oracle():
    config = ProbabilityProblemsConfig(seed=42, size=50)
    ds = ProbabilityProblemsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        score = ds.score_answer(item["answer"], item)
        assert score == 1.0, f"Item {i}: oracle scored {score}"


def test_score_none():
    config = ProbabilityProblemsConfig(seed=42, size=10)
    ds = ProbabilityProblemsDataset(config)
    item = ds[0]
    assert ds.score_answer(None, item) == 0.0


def test_score_wrong_answer():
    config = ProbabilityProblemsConfig(seed=42, size=10)
    ds = ProbabilityProblemsDataset(config)
    item = ds[0]
    assert ds.score_answer("not a fraction", item) == 0.0


def test_score_equivalent_fraction():
    config = ProbabilityProblemsConfig(seed=42, size=10, task_types=("independent_events",), task_weights=[1.0])
    ds = ProbabilityProblemsDataset(config)
    item = ds[0]
    oracle_frac = Fraction(item["answer"])
    unsimplified = f"{oracle_frac.numerator * 3}/{oracle_frac.denominator * 3}"
    score = ds.score_answer(unsimplified, item)
    assert score == 1.0


def test_curriculum():
    curriculum = ProbabilityProblemsCurriculum()
    base_value = {"size": 50, "seed": 1}
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1

    curriculum.increment_attr_level("n_range")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.max_n >= base_cfg.max_n


def test_task_types():
    for task_type in TASK_TYPES:
        config = ProbabilityProblemsConfig(seed=42, size=10, task_types=(task_type,), task_weights=[1.0])
        ds = ProbabilityProblemsDataset(config)
        for i in range(len(ds)):
            item = ds[i]
            assert item["metadata"]["task_type"] == task_type
            score = ds.score_answer(item["answer"], item)
            assert score == 1.0, f"Task {task_type}, item {i}: oracle scored {score}"


# --- Targeted tests for individual task types ---


def test_independent_events_math():
    config = ProbabilityProblemsConfig(seed=100, size=30, task_types=("independent_events",), task_weights=[1.0])
    ds = ProbabilityProblemsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        frac = Fraction(item["answer"])
        assert 0 < frac <= 1


def test_compound_events_math():
    config = ProbabilityProblemsConfig(seed=42, size=20, task_types=("compound_events",), task_weights=[1.0])
    ds = ProbabilityProblemsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        frac = Fraction(item["answer"])
        assert 0 < frac < 1


def test_total_probability_in_range():
    config = ProbabilityProblemsConfig(seed=42, size=20, task_types=("total_probability",), task_weights=[1.0])
    ds = ProbabilityProblemsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        frac = Fraction(item["answer"])
        assert 0 < frac < 1, f"Item {i}: P(red) = {frac} not in (0,1)"


def test_bayes_theorem_in_range():
    config = ProbabilityProblemsConfig(seed=42, size=20, task_types=("bayes_theorem",), task_weights=[1.0])
    ds = ProbabilityProblemsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        frac = Fraction(item["answer"])
        assert 0 < frac <= 1, f"Item {i}: P(Bag|red) = {frac} not in (0,1]"


def test_binomial_probability_in_range():
    config = ProbabilityProblemsConfig(seed=42, size=20, task_types=("binomial_probability",), task_weights=[1.0])
    ds = ProbabilityProblemsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        frac = Fraction(item["answer"])
        assert 0 < frac <= 1


def test_binomial_stats_positive():
    config = ProbabilityProblemsConfig(seed=42, size=20, task_types=("binomial_stats",), task_weights=[1.0])
    ds = ProbabilityProblemsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        frac = Fraction(item["answer"])
        assert frac > 0


def test_geometric_series_in_range():
    config = ProbabilityProblemsConfig(seed=42, size=20, task_types=("geometric_series",), task_weights=[1.0])
    ds = ProbabilityProblemsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        frac = Fraction(item["answer"])
        assert 0 < frac < 1, f"Item {i}: P(A wins) = {frac} not in (0,1)"


def test_geometric_series_manual():
    """With p=q=1/2: P(A wins) = (1/2)/(1 - 1/4) = 2/3."""
    config = ProbabilityProblemsConfig(seed=0, size=50, task_types=("geometric_series",), task_weights=[1.0])
    ds = ProbabilityProblemsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        if "1/2" in item["question"]:
            q = item["question"]
            if q.count("1/2") >= 2:
                assert item["answer"] == "2/3", f"With p=q=1/2, expected 2/3, got {item['answer']}"
                break


def test_geometric_region_in_range():
    config = ProbabilityProblemsConfig(seed=42, size=20, task_types=("geometric_region",), task_weights=[1.0])
    ds = ProbabilityProblemsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        frac = Fraction(item["answer"])
        assert 0 < frac <= Fraction(1, 2), f"Item {i}: region prob = {frac}"


def test_expectation_variance_types():
    config = ProbabilityProblemsConfig(seed=42, size=30, task_types=("expectation_variance",), task_weights=[1.0])
    ds = ProbabilityProblemsDataset(config)
    seen_exp = False
    seen_var = False
    for i in range(len(ds)):
        item = ds[i]
        frac = Fraction(item["answer"])
        if "E(X)" in item["question"]:
            assert frac > 0
            seen_exp = True
        if "Var(X)" in item["question"]:
            assert frac >= 0
            seen_var = True
    assert seen_exp and seen_var, "Should generate both expectation and variance problems"

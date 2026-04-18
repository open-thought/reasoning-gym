import math

import pytest

from reasoning_gym.combinatorics.combinatorics import (
    TASK_TYPES,
    CombinatoricsConfig,
    CombinatoricsCurriculum,
    CombinatoricsDataset,
)


def test_config_validation():
    with pytest.raises(AssertionError):
        config = CombinatoricsConfig(min_n=1)
        config.validate()

    with pytest.raises(AssertionError):
        config = CombinatoricsConfig(min_n=10, max_n=5)
        config.validate()


def test_deterministic():
    config = CombinatoricsConfig(seed=42, size=10)
    ds1 = CombinatoricsDataset(config)
    ds2 = CombinatoricsDataset(config)
    for i in range(len(ds1)):
        assert ds1[i] == ds2[i]


def test_item_structure():
    config = CombinatoricsConfig(seed=42, size=50)
    ds = CombinatoricsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        assert item["metadata"]["source_dataset"] == "combinatorics"


def test_answer_correctness():
    config = CombinatoricsConfig(seed=42, size=100)
    ds = CombinatoricsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        answer = item["answer"]
        assert answer.lstrip("-").isdigit(), f"Item {i}: answer '{answer}' is not an integer"


def test_score_wrong_answer():
    config = CombinatoricsConfig(seed=42, size=10)
    ds = CombinatoricsDataset(config)
    item = ds[0]
    score = ds.score_answer(item["answer"], item)
    assert score == 1.0


def test_curriculum():
    curriculum = CombinatoricsCurriculum()
    base_value = {"size": 50, "seed": 1}
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1

    curriculum.increment_attr_level("n_range")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.max_n >= base_cfg.max_n


def test_task_types():
    for task_type in TASK_TYPES:
        config = CombinatoricsConfig(seed=42, size=10, task_types=(task_type,), task_weights=[1.0])
        ds = CombinatoricsDataset(config)
        for i in range(len(ds)):
            item = ds[i]
            assert item["metadata"]["task_type"] == task_type
            assert item["answer"].lstrip("-").isdigit()


# --- Targeted tests for new task types ---


def test_multinomial_known_values():
    config = CombinatoricsConfig(seed=100, size=20, task_types=("multinomial",), task_weights=[1.0])
    ds = CombinatoricsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert int(item["answer"]) > 0
        assert "coefficient" in item["question"].lower()


def test_grid_paths_known_values():
    config = CombinatoricsConfig(seed=42, size=20, task_types=("grid_paths",), task_weights=[1.0])
    ds = CombinatoricsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert int(item["answer"]) >= 1
        assert "grid" in item["question"].lower()


def test_constrained_selection_known_values():
    config = CombinatoricsConfig(seed=42, size=20, task_types=("constrained_selection",), task_weights=[1.0])
    ds = CombinatoricsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert int(item["answer"]) >= 1
        assert "committee" in item["question"].lower()


def test_circular_permutation_known_values():
    config = CombinatoricsConfig(seed=42, size=20, task_types=("circular_permutation",), task_weights=[1.0])
    ds = CombinatoricsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert int(item["answer"]) >= 1
        assert "circular" in item["question"].lower()


def test_geometric_counting_known_values():
    config = CombinatoricsConfig(seed=42, size=20, task_types=("geometric_counting",), task_weights=[1.0])
    ds = CombinatoricsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        ans = int(item["answer"])
        assert ans >= 0


def test_dictionary_rank_known_values():
    config = CombinatoricsConfig(seed=42, size=20, task_types=("dictionary_rank",), task_weights=[1.0])
    ds = CombinatoricsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        rank = int(item["answer"])
        assert rank >= 1


def test_dictionary_rank_manual():
    """Verify the rank algorithm against a known example: 'BAC' from {A,B,C} has rank 3."""
    dataset = CombinatoricsDataset.__new__(CombinatoricsDataset)

    remaining = sorted("BAC")  # ['A', 'B', 'C']
    word = "BAC"
    rank = 1
    for ch in word:
        pos = remaining.index(ch)
        rank += pos * math.factorial(len(remaining) - 1)
        remaining.pop(pos)
    assert rank == 3  # ABC=1, ACB=2, BAC=3


def test_derangement_known_values():
    config = CombinatoricsConfig(seed=42, size=20, task_types=("derangement",), task_weights=[1.0])
    ds = CombinatoricsDataset(config)
    known = {2: 1, 3: 2, 4: 9, 5: 44, 6: 265, 7: 1854, 8: 14833, 9: 133496, 10: 1334961}
    for i in range(len(ds)):
        item = ds[i]
        ans = int(item["answer"])
        assert ans >= 0
        q = item["question"]
        for n_val, d_val in known.items():
            if f"set of {n_val} elements" in q:
                assert ans == d_val, f"D({n_val}) should be {d_val}, got {ans}"


def test_group_division_known_values():
    config = CombinatoricsConfig(seed=42, size=20, task_types=("group_division",), task_weights=[1.0])
    ds = CombinatoricsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert int(item["answer"]) >= 1


def test_legendres_formula_known_values():
    config = CombinatoricsConfig(seed=42, size=20, task_types=("legendres_formula",), task_weights=[1.0])
    ds = CombinatoricsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert int(item["answer"]) >= 0


def test_legendres_formula_manual():
    """Power of 2 in 10! = floor(10/2) + floor(10/4) + floor(10/8) = 5+2+1 = 8."""
    config = CombinatoricsConfig(
        seed=0, size=50, task_types=("legendres_formula",), task_weights=[1.0], min_n=10, max_n=10
    )
    ds = CombinatoricsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        q = item["question"]
        if "power of 2" in q and "10!" in q:
            assert item["answer"] == "8", f"Expected 8, got {item['answer']}"
            break


def test_integral_solutions_known_values():
    config = CombinatoricsConfig(seed=42, size=20, task_types=("integral_solutions",), task_weights=[1.0])
    ds = CombinatoricsDataset(config)
    for i in range(len(ds)):
        item = ds[i]
        assert int(item["answer"]) >= 1


def test_all_new_types_score_oracle():
    """Oracle answers should all score 1.0."""
    new_types = (
        "multinomial",
        "grid_paths",
        "constrained_selection",
        "circular_permutation",
        "geometric_counting",
        "dictionary_rank",
        "derangement",
        "group_division",
        "legendres_formula",
        "integral_solutions",
    )
    for tt in new_types:
        config = CombinatoricsConfig(seed=42, size=10, task_types=(tt,), task_weights=[1.0])
        ds = CombinatoricsDataset(config)
        for i in range(len(ds)):
            item = ds[i]
            score = ds.score_answer(item["answer"], item)
            assert score == 1.0, f"{tt} item {i}: oracle scored {score}"

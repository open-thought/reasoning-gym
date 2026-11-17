import random

import pytest

import reasoning_gym
from reasoning_gym.security import (
    AutonomyLevelControlConfig,
    AutonomyLevelControlCurriculum,
    AutonomyLevelControlDataset,
)
from reasoning_gym.security.autonomy_level_control import (
    ACTIONS,
    CRITICAL_ASSET_THRESHOLD,
    REFERENCE_POLICY,
    decide_level,
)


def test_autonomy_level_control_config_validation():
    """Test that invalid configs raise appropriate errors"""
    bad_configs = [
        dict(min_steps=0),
        dict(min_steps=4, max_steps=3),
        dict(min_rules=0),
        dict(min_rules=4, max_rules=3),
        dict(min_conditions=0),
        dict(min_conditions=3, max_conditions=2),
        dict(max_conditions=8),
        dict(distractor_prob=1.5),
    ]
    for kwargs in bad_configs:
        with pytest.raises(AssertionError):
            AutonomyLevelControlConfig(**kwargs).validate()


def test_autonomy_level_control_deterministic():
    """Test that dataset generates same items with same seed"""
    config = AutonomyLevelControlConfig(seed=42, size=20)
    dataset1 = AutonomyLevelControlDataset(config)
    dataset2 = AutonomyLevelControlDataset(config)
    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_autonomy_level_control_items():
    """Answers match an independent re-application of the stored policy, and items respect the config"""
    config = AutonomyLevelControlConfig(
        min_steps=2, max_steps=8, min_rules=2, max_rules=7, min_conditions=1, max_conditions=3, size=200, seed=7
    )
    dataset = AutonomyLevelControlDataset(config)
    for item in dataset:
        meta = item["metadata"]
        assert meta["source_dataset"] == "autonomy_level_control"
        assert config.min_steps <= meta["num_steps"] == len(meta["steps"]) <= config.max_steps
        assert config.min_rules <= meta["num_rules"] <= config.max_rules
        for rule in meta["policy"]["rules"]:
            assert config.min_conditions <= len(rule["conditions"]) <= config.max_conditions
            assert rule["level"] != "L5"
        assert meta["policy"]["default"] != "L5"

        decisions = [decide_level(meta["policy"], s) for s in meta["steps"]]
        assert item["answer"] == ", ".join(level for level, _ in decisions)
        assert meta["fired_rules"] == [pos for _, pos in decisions]

        # Fixed incident fields stay fixed across the timeline, and every step is shown.
        for i, step in enumerate(meta["steps"], 1):
            assert step["asset_criticality"] == meta["incident"]["asset_criticality"]
            assert step["threat_pattern"] == meta["incident"]["threat_pattern"]
            assert step["playbook"] == meta["incident"]["playbook"]
            assert f"Step {i}: threat_velocity={step['threat_velocity']}" in item["question"]
            assert f"agent_confidence={step['agent_confidence']:.2f}" in item["question"]
            assert (step["action"], step["action_reversible"] == "yes") in ACTIONS
        assert f"Rule {meta['num_rules']}: if" in item["question"]


def test_autonomy_level_control_first_match_semantics():
    """The earliest matching rule decides; boundaries are inclusive for >= and exclusive for <"""
    policy = {
        "rules": [
            {"conditions": [{"field": "telemetry_integrity", "op": "in", "value": ["DEGRADED"]}], "level": "L0"},
            {"conditions": [{"field": "agent_confidence", "op": ">=", "value": 0.8}], "level": "L4"},
            {"conditions": [{"field": "agent_confidence", "op": ">=", "value": 0.6}], "level": "L3"},
        ],
        "default": "L1",
    }
    base = {"telemetry_integrity": "OK"}
    assert decide_level(policy, {**base, "telemetry_integrity": "DEGRADED", "agent_confidence": 0.9}) == ("L0", 0)
    assert decide_level(policy, {**base, "agent_confidence": 0.8}) == ("L4", 1)
    assert decide_level(policy, {**base, "agent_confidence": 0.79}) == ("L3", 2)
    assert decide_level(policy, {**base, "agent_confidence": 0.59}) == ("L1", 3)


def test_autonomy_level_control_reference_policy():
    """The reference policy is used as given and never grants more than L1 on mission-critical assets"""
    dataset = AutonomyLevelControlDataset(AutonomyLevelControlConfig(use_reference_policy=True, size=100, seed=3))
    for item in dataset:
        assert item["metadata"]["policy"] == REFERENCE_POLICY
        assert item["metadata"]["reference_policy"] is True
        if item["metadata"]["incident"]["asset_criticality"] >= CRITICAL_ASSET_THRESHOLD:
            assert set(item["answer"].split(", ")) <= {"L0", "L1"}


def test_autonomy_level_control_timeline_varies():
    """Steps exercise different rules, so the level usually changes during the incident (sliding autonomy)"""
    dataset = AutonomyLevelControlDataset(AutonomyLevelControlConfig(min_steps=5, max_steps=5, size=100, seed=5))
    varied = sum(len(set(item["answer"].split(", "))) > 1 for item in dataset)
    assert varied >= 95
    fired = {pos for item in dataset for pos in item["metadata"]["fired_rules"]}
    assert len(fired) >= 4


def test_autonomy_level_control_distractors():
    """Distractors appear in the question but never change the answer"""
    with_d = AutonomyLevelControlDataset(AutonomyLevelControlConfig(distractor_prob=1.0, size=30, seed=9))
    for item in with_d:
        assert all(step["distractor"] for step in item["metadata"]["steps"])
        assert all(step["distractor"] in item["question"] for step in item["metadata"]["steps"])
        assert item["answer"] == ", ".join(
            decide_level(item["metadata"]["policy"], s)[0] for s in item["metadata"]["steps"]
        )
    without = AutonomyLevelControlDataset(AutonomyLevelControlConfig(distractor_prob=0.0, size=30, seed=9))
    assert all(step["distractor"] is None for item in without for step in item["metadata"]["steps"])


def _entry(dataset, critical: bool):
    for item in dataset:
        is_critical = item["metadata"]["incident"]["asset_criticality"] >= CRITICAL_ASSET_THRESHOLD
        levels = item["answer"].split(", ")
        if is_critical == critical and any(lvl != "L4" for lvl in levels) and any(lvl != "L0" for lvl in levels):
            return item
    raise AssertionError("no suitable entry found")


def test_autonomy_level_control_score_answer():
    dataset = AutonomyLevelControlDataset(AutonomyLevelControlConfig(min_steps=4, max_steps=4, size=200, seed=1))
    entry = _entry(dataset, critical=False)
    levels = entry["answer"].split(", ")

    assert dataset.score_answer(entry["answer"], entry) == 1.0
    assert dataset.score_answer("\n".join(lvl.lower() for lvl in levels) + ".", entry) == 1.0
    assert dataset.score_answer(None, entry) == 0.0
    assert dataset.score_answer("", entry) == 0.0
    assert dataset.score_answer(", ".join(levels[:-1]), entry) == 0.0  # wrong length
    assert dataset.score_answer(", ".join(["L5"] + levels[1:]), entry) == 0.0  # L5 is never permitted
    assert dataset.score_answer(", ".join(["X9"] + levels[1:]), entry) == 0.0  # not a level

    # One wrong (cautious) step out of four on a non-critical asset: half of 3/4.
    i = next(k for k, lvl in enumerate(levels) if lvl != "L0")
    cautious = levels.copy()
    cautious[i] = "L0"
    assert dataset.score_answer(", ".join(cautious), entry) == pytest.approx(0.375)


def test_autonomy_level_control_critical_overreach_scores_zero():
    dataset = AutonomyLevelControlDataset(AutonomyLevelControlConfig(min_steps=4, max_steps=4, size=200, seed=1))
    entry = _entry(dataset, critical=True)
    levels = entry["answer"].split(", ")
    i = next(k for k, lvl in enumerate(levels) if lvl != "L4")
    over, under = levels.copy(), levels.copy()
    over[i] = "L4"
    assert dataset.score_answer(", ".join(over), entry) == 0.0
    j = next(k for k, lvl in enumerate(levels) if lvl != "L0")
    under[j] = "L0"
    assert dataset.score_answer(", ".join(under), entry) == pytest.approx(0.375)


def test_autonomy_level_control_random_guessing_is_weak():
    dataset = AutonomyLevelControlDataset(AutonomyLevelControlConfig(size=300, seed=11))
    rng = random.Random(0)
    scores = [
        dataset.score_answer(
            ", ".join(rng.choice(["L0", "L1", "L2", "L3", "L4"]) for _ in item["metadata"]["steps"]), item
        )
        for item in dataset
    ]
    assert sum(scores) / len(scores) < 0.15


def test_autonomy_level_control_create_dataset():
    dataset = reasoning_gym.create_dataset("autonomy_level_control", size=5, seed=9)
    for entry in dataset:
        assert dataset.score_answer(entry["answer"], entry) == 1.0


def test_autonomy_level_control_curriculum():
    curriculum = AutonomyLevelControlCurriculum()
    base_value = {"size": 150, "seed": 1}

    base_cfg: AutonomyLevelControlConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1 and base_cfg.size == 150
    assert base_cfg.min_steps == 3 and base_cfg.max_steps == 5
    assert base_cfg.min_rules == 3 and base_cfg.max_rules == 5
    assert base_cfg.min_conditions == 1 and base_cfg.max_conditions == 2

    curriculum.increment_attr_level("num_steps")
    curriculum.increment_attr_level("num_rules")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_steps == 3 and increased_cfg.max_steps == 8
    assert increased_cfg.min_rules == 3 and increased_cfg.max_rules == 7

    curriculum.decrement_attr_level("num_rules")
    decreased_cfg = curriculum.generate_configuration(base_value)
    assert decreased_cfg.min_rules == 3 and decreased_cfg.max_rules == 5
    assert decreased_cfg.min_steps == 3 and decreased_cfg.max_steps == 8

    # The hardest level still generates valid items.
    for attr in ("num_steps", "num_rules", "num_conditions"):
        while curriculum.increment_attr_level(attr):
            pass
    hard_cfg = curriculum.generate_configuration({"size": 20, "seed": 2})
    for item in AutonomyLevelControlDataset(hard_cfg):
        assert len(item["answer"].split(", ")) == item["metadata"]["num_steps"]
        assert dataset_score_is_one(item)


def dataset_score_is_one(item) -> bool:
    return AutonomyLevelControlDataset(AutonomyLevelControlConfig(size=1)).score_answer(item["answer"], item) == 1.0

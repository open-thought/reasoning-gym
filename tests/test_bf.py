import re
from random import Random

import pytest

from reasoning_gym.code.bf import BFConfig, BFCurriculum, BFDataset


def test_bf():
    """Test basic properties and solution of generated items"""

    # Easy
    config = BFConfig(seed=42, size=20, difficulty=1)
    dataset = BFDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata contains required fields
        assert "bfit_code" in item["metadata"]
        assert "bf_program" in item["metadata"]

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
        assert dataset.score_answer(answer="Love is a battlefield", entry=item) == 0.0

    # Medium
    config = BFConfig(seed=43, size=20, difficulty=2)
    dataset = BFDataset(config)
    for item in dataset:
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0

    # Hard
    config = BFConfig(seed=44, size=20, difficulty=3)
    dataset = BFDataset(config)
    for item in dataset:
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0


def test_bf_curriculum():
    curriculum = BFCurriculum()

    base_value = {"size": 150, "seed": 1}

    base_cfg: BFConfig = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.difficulty == 1
    curriculum.increment_attr_level("difficulty")
    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.difficulty == 2


def _run_bfit_8bit(bfit_code: str):
    """Run the difficulty-3 loop with 8-bit cells. Returns acc, or None if it never halts."""

    def literal(pattern: str) -> int:
        return int(re.search(pattern, bfit_code).group(1)) % 256

    target = literal(r"int target = (\d+);")
    x = literal(r"int x = (\d+);")
    y = literal(r"int y = (\d+);")
    cond = literal(r"acc > (\d+)")
    acc = 0
    # acc is the only state, so 257 iterations without halting means a cycle
    for _ in range(257):
        if acc >= target:
            return acc
        acc = (acc + x) % 256
        if acc > cond:
            acc = (acc + y) % 256
    return None


def _difficulty_3_code(dataset: BFDataset, idx: int) -> str:
    return dataset.generate_bfit_code(3, Random(dataset.seed + idx))


def test_bf_difficulty_3_seed_12345_item_289_halts():
    dataset = BFDataset(BFConfig(seed=12345, size=500, difficulty=3))
    expected = _run_bfit_8bit(_difficulty_3_code(dataset, 289))
    # check before dataset[289], which hangs when the program never halts
    assert expected is not None
    assert dataset[289]["answer"] == str(expected)


def test_bf_difficulty_3_programs_halt():
    dataset = BFDataset(BFConfig(seed=12345, size=2000, difficulty=3))
    hanging = [i for i in range(len(dataset)) if _run_bfit_8bit(_difficulty_3_code(dataset, i)) is None]
    assert hanging == []


def test_bf_difficulty_3_answers_match_8bit_run():
    dataset = BFDataset(BFConfig(seed=7, size=10, difficulty=3))
    for i in range(len(dataset)):
        expected = _run_bfit_8bit(_difficulty_3_code(dataset, i))
        assert expected is not None
        assert dataset[i]["answer"] == str(expected)

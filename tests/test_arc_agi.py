import pytest

from reasoning_gym.arc.arc_agi import ArcAgiConfig, ArcAgiDataset


def test_arc_agi_config_validation():
    """Test validation of ArcAgi configuration parameters"""
    with pytest.raises(AssertionError):
        ArcAgiConfig(size=0).validate()

    # Valid config should not raise
    config = ArcAgiConfig(size=10, seed=42)
    config.validate()


def test_arc_agi_deterministic():
    """Test dataset reproducibility with fixed seed"""
    config = ArcAgiConfig(seed=42, size=10)
    ds1 = ArcAgiDataset(config)
    ds2 = ArcAgiDataset(config)

    for i in range(len(ds1)):
        assert ds1[i] == ds2[i], "ArcAgi datasets with same seed should match exactly"


def test_arc_agi_items():
    """Test basic structure and metadata of generated items"""
    config = ArcAgiConfig(seed=42, size=10)
    dataset = ArcAgiDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        meta = item["metadata"]
        assert "input" in meta
        assert "output" in meta
        assert "task_id" in meta

        # Verify input/output are tuples of tuples (board format)
        assert isinstance(meta["input"], tuple)
        assert isinstance(meta["output"], tuple)
        assert all(isinstance(row, tuple) for row in meta["input"])
        assert all(isinstance(row, tuple) for row in meta["output"])

        # Verify task_id is a string
        assert isinstance(meta["task_id"], str)


def test_arc_agi_scoring():
    """Test solution verification and scoring"""
    config = ArcAgiConfig(size=10, seed=123)
    dataset = ArcAgiDataset(config)

    for item in dataset:
        # Test correct solution
        assert dataset.score_answer(item["answer"], entry=item) == 1.0

        # Test invalid format
        assert dataset.score_answer("invalid grid format", entry=item) == 0.01

        # Test None answer
        assert dataset.score_answer(None, entry=item) == 0.0

        # Test wrong but valid grid format
        wrong_answer = "0 0\n0 0"
        assert dataset.score_answer(wrong_answer, entry=item) == 0.05


def test_arc_agi_dataset_modes():
    """Test dataset behavior with different train/eval configurations"""
    # Test train-only mode
    train_config = ArcAgiConfig(use_train=True, use_eval=False, size=10, seed=42)
    train_ds = ArcAgiDataset(train_config)
    assert len(train_ds._task_ids) > 0

    # Test eval-only mode
    eval_config = ArcAgiConfig(use_train=False, use_eval=True, size=10, seed=42)
    eval_ds = ArcAgiDataset(eval_config)
    assert len(eval_ds._task_ids) > 0

    # Test both modes
    both_config = ArcAgiConfig(use_train=True, use_eval=True, size=10, seed=42)
    both_ds = ArcAgiDataset(both_config)
    assert len(both_ds._task_ids) > len(train_ds._task_ids)
    assert len(both_ds._task_ids) > len(eval_ds._task_ids)

"""Unit tests for the fund_ratios finance dataset.

Mirrors the structure of tests/test_chain_sum.py: config validation,
determinism, item contract, value ranges, per-metric numeric correctness,
TVPI additivity, the MOIC worked example, scoring behavior, and curriculum
behavior.
"""

from decimal import ROUND_HALF_UP, Decimal

import pytest

import reasoning_gym
from reasoning_gym.finance import FundRatiosConfig, FundRatiosCurriculum, FundRatiosDataset
from reasoning_gym.finance.fund_ratios import DATASET_NAME, SUPPORTED_METRICS, FundFigures, compute_ratio


def test_fund_ratios_config_validation():
    """Every invalid config branch raises AssertionError."""
    # Valid default config should not raise.
    FundRatiosConfig().validate()

    # size <= 0
    with pytest.raises(AssertionError):
        FundRatiosConfig(size=0).validate()
    with pytest.raises(AssertionError):
        FundRatiosConfig(size=-1).validate()

    # Inverted ranges for every configurable pair.
    with pytest.raises(AssertionError):
        FundRatiosConfig(min_commitment=500, max_commitment=100).validate()
    with pytest.raises(AssertionError):
        FundRatiosConfig(min_called_fraction=0.9, max_called_fraction=0.5).validate()
    with pytest.raises(AssertionError):
        FundRatiosConfig(min_expense_fraction=0.5, max_expense_fraction=0.1).validate()
    with pytest.raises(AssertionError):
        FundRatiosConfig(min_distribution_multiple=2.0, max_distribution_multiple=1.0).validate()
    with pytest.raises(AssertionError):
        FundRatiosConfig(min_nav_multiple=2.0, max_nav_multiple=1.0).validate()

    # Non-positive minimum commitment.
    with pytest.raises(AssertionError):
        FundRatiosConfig(min_commitment=0, max_commitment=100).validate()
    with pytest.raises(AssertionError):
        FundRatiosConfig(min_commitment=-10, max_commitment=100).validate()

    # Called-capital fraction outside (0, 1].
    with pytest.raises(AssertionError):
        FundRatiosConfig(min_called_fraction=0.0).validate()
    with pytest.raises(AssertionError):
        FundRatiosConfig(min_called_fraction=-0.1).validate()
    with pytest.raises(AssertionError):
        FundRatiosConfig(max_called_fraction=1.5).validate()

    # Expense fraction outside [0, 1).
    with pytest.raises(AssertionError):
        FundRatiosConfig(min_expense_fraction=-0.1).validate()
    with pytest.raises(AssertionError):
        FundRatiosConfig(max_expense_fraction=1.0).validate()
    with pytest.raises(AssertionError):
        FundRatiosConfig(max_expense_fraction=1.5).validate()

    # Non-negative multiples.
    with pytest.raises(AssertionError):
        FundRatiosConfig(min_distribution_multiple=-0.1).validate()
    with pytest.raises(AssertionError):
        FundRatiosConfig(min_nav_multiple=-0.1).validate()

    # Negative decimal places.
    with pytest.raises(AssertionError):
        FundRatiosConfig(decimal_places=-1).validate()

    # Empty metrics / unknown metric.
    with pytest.raises(AssertionError):
        FundRatiosConfig(metrics=()).validate()
    with pytest.raises(AssertionError):
        FundRatiosConfig(metrics=("dpi", "bogus")).validate()


def test_fund_ratios_deterministic():
    """Two datasets with the same seed/size produce identical items."""
    config = FundRatiosConfig(seed=42, size=20)
    dataset1 = FundRatiosDataset(config)
    dataset2 = FundRatiosDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_fund_ratios_item_contract():
    """Each item has question/answer/metadata with required keys."""
    config = FundRatiosConfig(seed=123, size=50)
    dataset = FundRatiosDataset(config)

    required_metadata_keys = {
        "source_dataset",
        "source_index",
        "metric",
        "commitment",
        "called",
        "distributions",
        "nav",
        "expenses",
        "result",
        "decimal_places",
        "difficulty",
    }

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        assert isinstance(item["question"], str) and len(item["question"]) > 0
        assert isinstance(item["answer"], str) and len(item["answer"]) > 0

        meta = item["metadata"]
        assert required_metadata_keys.issubset(meta.keys())
        assert meta["source_dataset"] == DATASET_NAME
        assert meta["source_index"] == i
        assert meta["metric"] in SUPPORTED_METRICS

        # difficulty sub-dict shape
        assert "commitment" in meta["difficulty"]
        assert "decimal_places" in meta["difficulty"]

        # answer must be parseable as a Decimal
        Decimal(item["answer"])


def test_fund_ratios_iteration():
    """Iteration yields exactly `size` items and is repeatable."""
    config = FundRatiosConfig(seed=7, size=5)
    dataset = FundRatiosDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    first = list(dataset)
    second = list(dataset)
    assert first == second


def test_fund_ratios_value_ranges():
    """Across many items 0 < called <= commitment and 0 <= expenses < called."""
    config = FundRatiosConfig(seed=99, size=200)
    dataset = FundRatiosDataset(config)

    for i in range(len(dataset)):
        meta = dataset[i]["metadata"]
        commitment = meta["commitment"]
        called = meta["called"]
        expenses = meta["expenses"]
        distributions = meta["distributions"]
        nav = meta["nav"]

        assert 0 < called <= commitment, f"called {called} not in (0, {commitment}]"
        assert 0 <= expenses < called, f"expenses {expenses} not in [0, {called})"
        assert distributions >= 0
        assert nav >= 0

        # commitment within configured bounds (allowing for rounding to whole units)
        assert config.min_commitment - 1 <= commitment <= config.max_commitment + 1


def _recompute(metric: str, meta: dict) -> Decimal:
    """Independently recompute the exact ratio from metadata figures."""
    called = Decimal(str(meta["called"]))
    distributions = Decimal(str(meta["distributions"]))
    nav = Decimal(str(meta["nav"]))
    expenses = Decimal(str(meta["expenses"]))
    total_value = nav + distributions

    if metric == "dpi":
        return distributions / called
    if metric == "rvpi":
        return nav / called
    if metric == "tvpi":
        return total_value / called
    if metric == "moic":
        return (total_value - expenses) / (called - expenses)
    raise ValueError(metric)


@pytest.mark.parametrize("metric", SUPPORTED_METRICS)
def test_fund_ratios_numeric_correctness(metric):
    """Per-metric: recompute the ratio independently and compare to stored answer."""
    config = FundRatiosConfig(seed=2024, size=100, metrics=(metric,), decimal_places=2)
    dataset = FundRatiosDataset(config)

    quantum = Decimal(1).scaleb(-config.decimal_places)

    for i in range(len(dataset)):
        item = dataset[i]
        meta = item["metadata"]
        assert meta["metric"] == metric

        expected_exact = _recompute(metric, meta)
        expected_rounded = expected_exact.quantize(quantum, rounding=ROUND_HALF_UP)

        assert Decimal(item["answer"]) == expected_rounded


def test_fund_ratios_tvpi_additivity():
    """TVPI == DPI + RVPI for identical underlying figures.

    Single-metric configs sharing the same seed/size generate identical
    figures per index, so comparing answers index-by-index is valid.
    """
    size = 100
    seed = 555
    dp = 4  # extra precision so the additive identity holds exactly after rounding

    dpi_ds = FundRatiosDataset(FundRatiosConfig(seed=seed, size=size, metrics=("dpi",), decimal_places=dp))
    rvpi_ds = FundRatiosDataset(FundRatiosConfig(seed=seed, size=size, metrics=("rvpi",), decimal_places=dp))
    tvpi_ds = FundRatiosDataset(FundRatiosConfig(seed=seed, size=size, metrics=("tvpi",), decimal_places=dp))

    quantum = Decimal(1).scaleb(-dp)
    one_ulp = quantum  # one unit in the last rounded place

    for i in range(size):
        dpi_meta = dpi_ds[i]["metadata"]
        rvpi_meta = rvpi_ds[i]["metadata"]
        tvpi_meta = tvpi_ds[i]["metadata"]

        # Sanity: the single-metric datasets really share the same figures.
        assert dpi_meta["called"] == rvpi_meta["called"] == tvpi_meta["called"]
        assert dpi_meta["distributions"] == tvpi_meta["distributions"]
        assert rvpi_meta["nav"] == tvpi_meta["nav"]

        dpi = Decimal(dpi_ds[i]["answer"])
        rvpi = Decimal(rvpi_ds[i]["answer"])
        tvpi = Decimal(tvpi_ds[i]["answer"])

        assert abs(tvpi - (dpi + rvpi)) <= one_ulp


def test_fund_ratios_moic_worked_example():
    """MOIC 100M / 80M called / 20M expenses -> denominator 60M.

    Uses the helper functions directly rather than relying on random draws.
    commitment=100M, called=80M, expenses=20M, distributions=90M, nav=30M.
    total_value = 120M; numerator = 120M - 20M = 100M; denominator = 80M - 20M = 60M.
    MOIC = 100/60 = 1.6667 (rounded to 2dp => 1.67).
    """
    figures = FundFigures(
        commitment=Decimal("100000000"),
        called=Decimal("80000000"),
        expenses=Decimal("20000000"),
        distributions=Decimal("90000000"),
        nav=Decimal("30000000"),
    )

    # Denominator is strictly positive and equals 60M.
    denominator = figures.called - figures.expenses
    assert denominator == Decimal("60000000")

    exact, rounded = compute_ratio("moic", figures, decimal_places=2)

    total_value = figures.nav + figures.distributions
    expected_exact = (total_value - figures.expenses) / denominator
    assert exact == expected_exact
    assert exact == Decimal("100000000") / Decimal("60000000")
    assert rounded == Decimal("1.67")

    # Cross-check the other metrics on the same figures for good measure.
    _, dpi = compute_ratio("dpi", figures, 2)
    _, rvpi = compute_ratio("rvpi", figures, 2)
    _, tvpi = compute_ratio("tvpi", figures, 2)
    assert dpi == Decimal("1.13")  # 90/80 = 1.125 -> 1.13 (ROUND_HALF_UP)
    assert rvpi == Decimal("0.38")  # 30/80 = 0.375 -> 0.38
    assert tvpi == Decimal("1.50")  # 120/80 = 1.50


def test_fund_ratios_scoring():
    """Exact, tolerant formats score 1.0; wrong and empty score 0.0."""
    config = FundRatiosConfig(seed=11, size=1)
    dataset = FundRatiosDataset(config)
    entry = {"answer": "1.50"}

    # Exact match.
    assert dataset.score_answer("1.50", entry) == 1.0
    # Tolerant numeric-equal formats.
    assert dataset.score_answer("1.5", entry) == 1.0
    assert dataset.score_answer("1.5x", entry) == 1.0
    assert dataset.score_answer("1.5X", entry) == 1.0
    assert dataset.score_answer("+1.5", entry) == 1.0
    assert dataset.score_answer("  1.50  ", entry) == 1.0

    # Commas with a larger oracle value.
    entry_big = {"answer": "1500.00"}
    assert dataset.score_answer("1,500", entry_big) == 1.0

    # Numerically wrong.
    assert dataset.score_answer("2.00", entry) == 0.0

    # None / empty.
    assert dataset.score_answer(None, entry) == 0.0
    assert dataset.score_answer("", entry) == 0.0


def test_fund_ratios_factory_create():
    """create_dataset returns a working dataset registered under the name."""
    dataset = reasoning_gym.create_dataset(DATASET_NAME, size=10, seed=3)
    assert isinstance(dataset, FundRatiosDataset)
    items = list(dataset)
    assert len(items) == 10
    for item in items:
        assert set(("question", "answer", "metadata")).issubset(item.keys())


def test_fund_ratios_curriculum():
    """Base config valid; incrementing commitment and decimal_places changes config."""
    curriculum = FundRatiosCurriculum()

    base_value = {"size": 150, "seed": 1}
    base_cfg: FundRatiosConfig = curriculum.generate_configuration(base_value)

    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    # Base level collapses the commitment range onto the smallest magnitude.
    assert base_cfg.min_commitment == 1_000_000
    assert base_cfg.max_commitment == 1_000_000
    assert base_cfg.decimal_places == 2
    # Base configuration must be valid.
    base_cfg.validate()

    # Increment commitment magnitude and decimal places.
    curriculum.increment_attr_level("commitment")
    curriculum.increment_attr_level("decimal_places")

    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_commitment == 1_000_000
    assert increased_cfg.max_commitment == 10_000_000
    assert increased_cfg.decimal_places == 3
    increased_cfg.validate()

    # Decrement decimal places back to base.
    curriculum.decrement_attr_level("decimal_places")
    partial_cfg = curriculum.generate_configuration(base_value)
    assert partial_cfg.decimal_places == 2
    assert partial_cfg.max_commitment == 10_000_000

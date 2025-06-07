import numpy as np
import pytest

from reasoning_gym.coaching.base_curriculum import DefaultCurriculumContext, RangeAttributeMode
from reasoning_gym.games import SurvoConfig, SurvoCurriculum, SurvoDataset


def test_survo_config_validation():
    """Bad configs should raise."""
    # min_board_size must be > 3
    with pytest.raises(AssertionError):
        SurvoConfig(min_board_size=3).validate()

    # max_board_size must be ≥ min_board_size
    with pytest.raises(AssertionError):
        SurvoConfig(min_board_size=6, max_board_size=5).validate()

    # min_empty ≤ max_empty and within board-area limits
    with pytest.raises(AssertionError):
        SurvoConfig(min_empty=6, max_empty=5).validate()

    # min_num < max_num
    with pytest.raises(AssertionError):
        SurvoConfig(min_num=5, max_num=5).validate()


def test_survo_deterministic():
    """Same seed ⇒ identical items."""
    cfg = SurvoConfig(seed=123, size=15, min_board_size=4, max_board_size=5, min_empty=3, max_empty=5)
    ds1, ds2 = SurvoDataset(cfg), SurvoDataset(cfg)

    for i in range(len(ds1)):
        assert ds1[i] == ds2[i]


def test_survo_items():
    """Generated items have expected structure and metadata."""
    cfg = SurvoConfig(seed=99, size=20, min_board_size=4, max_board_size=5, min_empty=3, max_empty=5)
    ds = SurvoDataset(cfg)

    for itm in ds:
        md = itm["metadata"]

        # Basic keys
        assert set(itm.keys()) == {"question", "answer", "metadata"}
        assert "puzzle" in md and "solution" in md and "candidate_numbers" in md

        orig = np.array(md["puzzle"])
        full = np.array(md["solution"])

        # Dimensions
        n = full.shape[0]
        assert cfg.min_board_size <= n <= cfg.max_board_size
        assert orig.shape == full.shape == (n, n)

        # Number of empties
        empties = np.count_nonzero(orig == 0)
        assert empties == md["num_empty"]
        assert cfg.min_empty <= empties <= cfg.max_empty

        # Candidate numbers should match removed values (order disregarded)
        removed_values = full[orig == 0].tolist()
        assert sorted(removed_values) == sorted(md["candidate_numbers"])


def test_survo_solution_validity():
    """Solution must satisfy Survo row/column-sum rules."""
    cfg = SurvoConfig(seed=321, size=10, min_board_size=4, max_board_size=5, min_empty=3, max_empty=5)
    ds = SurvoDataset(cfg)

    for itm in ds:
        m = np.array(itm["metadata"]["solution"])
        n = m.shape[0]

        # Row sums (exclude last row)
        for r in range(n - 1):
            assert m[r, : n - 1].sum() == m[r, n - 1]

        # Column sums (exclude last col)
        for c in range(n - 1):
            assert m[: n - 1, c].sum() == m[n - 1, c]

        # Grand total cell
        assert m[n - 1, n - 1] == m[: n - 1, n - 1].sum()


def test_survo_difficulty_levels():
    """More allowed empties ⇒ puzzles have, on average, more blank cells."""
    seed, n_items, board = 777, 8, 5

    def avg_empties(min_empty, max_empty):
        cfg = SurvoConfig(
            seed=seed,
            size=n_items,
            min_board_size=board,
            max_board_size=board,
            min_empty=min_empty,
            max_empty=max_empty,
        )
        ds = SurvoDataset(cfg)
        return np.mean([np.count_nonzero(np.array(itm["metadata"]["puzzle"]) == 0) for itm in ds])

    low = avg_empties(3, 3)
    mid = avg_empties(6, 6)
    high = avg_empties(10, 10)

    assert low < mid < high


def test_survo_answer_scoring():
    """Correct answer ⇒ 1.0; variations score lower."""
    cfg = SurvoConfig(seed=42, size=5, min_board_size=4, max_board_size=4, min_empty=3, max_empty=3)
    ds = SurvoDataset(cfg)

    for itm in ds:
        correct = itm["answer"]
        assert ds.score_answer(correct, itm) == 1.0

        # Tamper with a single cell
        candidate_numbers = itm["metadata"]["candidate_numbers"]
        wrong = correct.replace(str(candidate_numbers[0]), str(max(candidate_numbers) + 1), 1)
        assert ds.score_answer(wrong, itm) < 1.0

        # Bad type / empty
        assert ds.score_answer(None, itm) == 0.0
        assert ds.score_answer("", itm) == 0.0


def test_survo_curriculum():
    """SurvoCurriculum controls board size and empties as advertised."""
    cur = SurvoCurriculum()
    base_val = {"size": 100, "seed": 1}
    ctx = DefaultCurriculumContext(mode=RangeAttributeMode.UPPER_BOUND)

    # Level 0
    cfg0: SurvoConfig = cur.generate_configuration(base_val, context=ctx)
    assert cfg0.min_board_size == cfg0.max_board_size == 4
    assert cfg0.min_empty == cfg0.max_empty == 4

    # Increment levels
    cur.increment_attr_level("board_size")
    cur.increment_attr_level("empty")
    cfg1: SurvoConfig = cur.generate_configuration(base_val, context=ctx)
    assert cfg1.min_board_size == cfg1.max_board_size == 5
    assert cfg1.min_empty == cfg1.max_empty == 9

    # Global progression
    cur.set_global_level(3)
    cfg_max: SurvoConfig = cur.generate_configuration(base_val, context=ctx)
    assert cfg_max.min_board_size == cfg_max.max_board_size == 7
    assert cfg_max.min_empty == cfg_max.max_empty == 25

import itertools

import pytest

import reasoning_gym
from reasoning_gym.coaching.base_curriculum import DefaultCurriculumContext, RangeAttributeMode
from reasoning_gym.games import NonogramConfig, NonogramCurriculum, NonogramDataset
from reasoning_gym.games.nonogram import _line_possibilities, compute_clues, compute_line_clue, count_solutions


def test_nonogram_config_validation():
    """Invalid configs raise AssertionError"""
    for bad in (
        dict(size=0),
        dict(min_rows=1),
        dict(min_rows=6, max_rows=5),
        dict(min_cols=6, max_cols=5),
        dict(max_rows=26),
        dict(fill_prob=0.0),
        dict(fill_prob=1.0),
        dict(max_retries=0),
        dict(max_search_nodes=0),
    ):
        with pytest.raises(AssertionError):
            NonogramConfig(**bad).validate()


def test_nonogram_registered():
    ds = reasoning_gym.create_dataset("nonogram", size=5, seed=7)
    assert isinstance(ds, NonogramDataset)
    assert ds.category == "games"


def test_nonogram_deterministic():
    config = NonogramConfig(seed=42, size=10, min_rows=3, max_rows=8, min_cols=3, max_cols=8)
    ds1, ds2 = NonogramDataset(config), NonogramDataset(config)
    for i in range(len(ds1)):
        assert ds1[i] == ds2[i]


def test_compute_line_clue():
    assert compute_line_clue([0, 0, 0]) == []
    assert compute_line_clue([1, 1, 1]) == [3]
    assert compute_line_clue([1, 0, 1, 1, 0, 0, 1]) == [1, 2, 1]
    assert compute_line_clue([0, 1, 1, 0]) == [2]


def test_line_solver_matches_brute_force():
    """The DP line solver must agree with exhaustive enumeration."""
    for n in range(1, 7):
        for truth in itertools.product([0, 1], repeat=n):
            clue = tuple(compute_line_clue(truth))
            for known in itertools.product([-1, 0, 1], repeat=n):
                cands = [
                    c
                    for c in itertools.product([0, 1], repeat=n)
                    if tuple(compute_line_clue(c)) == clue and all(k == -1 or k == v for k, v in zip(known, c))
                ]
                res = _line_possibilities(known, clue)
                if not cands:
                    assert res is None
                else:
                    can_fill = [any(c[i] == 1 for c in cands) for i in range(n)]
                    can_empty = [any(c[i] == 0 for c in cands) for i in range(n)]
                    assert res == (can_fill, can_empty)


def test_count_solutions_matches_brute_force():
    for r, c in [(2, 2), (2, 3), (3, 3)]:
        for bits in itertools.product([0, 1], repeat=r * c):
            grid = [list(bits[i * c : (i + 1) * c]) for i in range(r)]
            row_clues, col_clues = compute_clues(grid)
            brute = sum(
                1
                for other in itertools.product([0, 1], repeat=r * c)
                if compute_clues([list(other[i * c : (i + 1) * c]) for i in range(r)]) == (row_clues, col_clues)
            )
            assert count_solutions(row_clues, col_clues, limit=10**6, max_nodes=None) == brute


def test_count_solutions_known_ambiguous():
    # The two diagonals of a 2x2 grid share the clues [1],[1] / [1],[1]
    assert count_solutions([[1], [1]], [[1], [1]], limit=10) == 2
    assert count_solutions([[2], [2]], [[2], [2]]) == 1


def test_nonogram_items():
    config = NonogramConfig(seed=42, size=20, min_rows=3, max_rows=10, min_cols=4, max_cols=9, fill_prob=0.55)
    ds = NonogramDataset(config)
    for item in ds:
        assert set(item.keys()) >= {"question", "answer", "metadata"}
        md = item["metadata"]
        assert md["source_dataset"] == "nonogram"
        assert config.min_rows <= md["n_rows"] <= config.max_rows
        assert config.min_cols <= md["n_cols"] <= config.max_cols
        solution = md["solution"]
        assert len(solution) == md["n_rows"] and all(len(row) == md["n_cols"] for row in solution)
        assert all(cell in (0, 1) for row in solution for cell in row)
        # Clues are consistent with the hidden solution
        assert compute_clues(solution) == (md["row_clues"], md["col_clues"])
        # The question shows every clue line and the answer encodes the solution
        assert item["question"].count("Row ") >= md["n_rows"]
        assert item["question"].count("Column ") >= md["n_cols"]
        assert item["answer"] == "\n".join(" ".join(map(str, row)) for row in solution)
        assert md["difficulty"] == {
            "rows": (config.min_rows, config.max_rows),
            "cols": (config.min_cols, config.max_cols),
            "fill_prob": config.fill_prob,
        }


def test_nonogram_unique_solutions():
    config = NonogramConfig(seed=3, size=15, min_rows=4, max_rows=10, min_cols=4, max_cols=10, fill_prob=0.55)
    for item in NonogramDataset(config):
        md = item["metadata"]
        assert md["unique_solution"]
        assert count_solutions(md["row_clues"], md["col_clues"], limit=2, max_nodes=None) == 1


def test_nonogram_non_unique_mode_flags_correctly():
    config = NonogramConfig(seed=5, size=30, min_rows=3, max_rows=4, min_cols=3, max_cols=4, require_unique=False)
    for item in NonogramDataset(config):
        md = item["metadata"]
        n = count_solutions(md["row_clues"], md["col_clues"], limit=2, max_nodes=None)
        assert md["unique_solution"] == (n == 1)


def test_nonogram_score_answer():
    ds = NonogramDataset(NonogramConfig(seed=11, size=10, min_rows=4, max_rows=8, min_cols=4, max_cols=8))
    for item in ds:
        answer = item["answer"]
        md = item["metadata"]
        assert ds.score_answer(answer, item) == 1.0

        # Lenient formats: #/. symbols, no separators, code fences, row labels
        sym = answer.replace("1", "#").replace("0", ".")
        assert ds.score_answer(sym, item) == 1.0
        assert ds.score_answer(sym.replace(" ", ""), item) == 1.0
        assert ds.score_answer("```\n" + answer + "\n```", item) == 1.0
        labelled = "\n".join(f"Row {i + 1}: {line}" for i, line in enumerate(answer.splitlines()))
        assert ds.score_answer(labelled, item) == 1.0

        # Flipping any single cell breaks at least its row clue
        grid = [row[:] for row in md["solution"]]
        grid[0][0] ^= 1
        flipped = "\n".join(" ".join(map(str, row)) for row in grid)
        assert ds.score_answer(flipped, item) == 0.0

        # Wrong shape / empty / None
        assert ds.score_answer("\n".join(answer.splitlines()[:-1]), item) == 0.0
        assert ds.score_answer(answer + "\n" + answer.splitlines()[0], item) == 0.0
        assert ds.score_answer("", item) == 0.0
        assert ds.score_answer(None, item) == 0.0


def test_nonogram_score_accepts_alternative_solution():
    """Any grid satisfying all clues earns full reward, even if it differs from the stored solution."""
    ds = NonogramDataset(NonogramConfig(seed=0, size=1, min_rows=2, max_rows=2, min_cols=2, max_cols=2))
    entry = {
        "question": "",
        "answer": "1 0\n0 1",
        "metadata": {"n_rows": 2, "n_cols": 2, "row_clues": [[1], [1]], "col_clues": [[1], [1]]},
    }
    assert ds.score_answer("1 0\n0 1", entry) == 1.0
    assert ds.score_answer("0 1\n1 0", entry) == 1.0
    assert ds.score_answer("1 1\n0 0", entry) == 0.0


def test_nonogram_curriculum():
    curriculum = NonogramCurriculum()
    base_value = {"size": 50, "seed": 1}
    context = DefaultCurriculumContext(mode=RangeAttributeMode.UPPER_BOUND)

    cfg: NonogramConfig = curriculum.generate_configuration(base_value, context=context)
    assert cfg.seed == 1 and cfg.size == 50
    assert cfg.min_rows == cfg.max_rows == 4
    assert cfg.min_cols == cfg.max_cols == 4
    assert cfg.fill_prob == 0.7

    curriculum.increment_attr_level("rows")
    curriculum.increment_attr_level("fill_prob")
    cfg = curriculum.generate_configuration(base_value, context=context)
    assert cfg.min_rows == cfg.max_rows == 5
    assert cfg.min_cols == cfg.max_cols == 4
    assert cfg.fill_prob == 0.6

    for _ in range(10):  # saturates at the top level
        curriculum.increment_attr_level("fill_prob")
    cfg = curriculum.generate_configuration(base_value, context=context)
    assert cfg.fill_prob == 0.5

    curriculum.decrement_attr_level("fill_prob")
    assert curriculum.generate_configuration(base_value, context=context).fill_prob == 0.55

    curriculum.set_global_level(0)
    cfg = curriculum.generate_configuration(base_value, context=context)
    assert cfg.min_rows == 4 and cfg.fill_prob == 0.7
    curriculum.increment_global_level()
    cfg = curriculum.generate_configuration(base_value, context=context)
    assert cfg.min_rows == 5 and cfg.min_cols == 5 and cfg.fill_prob == 0.6

    # Hardest level still generates valid, quickly-produced puzzles
    curriculum.set_global_level(len(curriculum.attributes["rows"].levels) - 1)
    hard_cfg = curriculum.generate_configuration({"size": 3, "seed": 1}, context=context)
    hard_ds = NonogramDataset(hard_cfg)
    for item in hard_ds:
        assert hard_ds.score_answer(item["answer"], item) == 1.0

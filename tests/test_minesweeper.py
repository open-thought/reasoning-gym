import pytest

from reasoning_gym.games.minesweeper import MinesweeperConfig, MinesweeperDataset, count_solutions, neighbors


def test_config_validation():
    bad_configs = [
        dict(min_rows=1),
        dict(min_rows=7, max_rows=5),
        dict(min_cols=1),
        dict(mine_density=0.0),
        dict(mine_density=0.6),
        dict(hide_safe_prob=1.0),
        dict(max_attempts=0),
    ]
    for kwargs in bad_configs:
        with pytest.raises(AssertionError):
            MinesweeperConfig(**kwargs).validate()


def test_deterministic():
    a = MinesweeperDataset(MinesweeperConfig(seed=42, size=10))
    b = MinesweeperDataset(MinesweeperConfig(seed=42, size=10))
    for i in range(len(a)):
        assert a[i] == b[i]


def test_different_seeds_differ():
    a = MinesweeperDataset(MinesweeperConfig(seed=1, size=5))
    b = MinesweeperDataset(MinesweeperConfig(seed=2, size=5))
    assert any(a[i]["question"] != b[i]["question"] for i in range(5))


def test_item_structure():
    ds = MinesweeperDataset(MinesweeperConfig(seed=42, size=10))
    for item in ds:
        assert set(item.keys()) == {"question", "answer", "metadata"}
        assert isinstance(item["question"], str)
        assert isinstance(item["answer"], str)
        md = item["metadata"]
        assert md["num_mines"] == len(md["mines"]) >= 1
        assert len(md["board"]) == md["rows"]
        assert all(len(row) == md["cols"] for row in md["board"])


def test_board_is_consistent_and_unique():
    ds = MinesweeperDataset(MinesweeperConfig(seed=7, size=15))
    for item in ds:
        md = item["metadata"]
        rows, cols = md["rows"], md["cols"]
        mines = {tuple(m) for m in md["mines"]}
        clues, hidden = {}, []
        for r in range(rows):
            for c in range(cols):
                cell = md["board"][r][c]
                if cell == "?":
                    hidden.append((r, c))
                else:
                    assert (r, c) not in mines
                    expected = sum(1 for n in neighbors(r, c, rows, cols) if n in mines)
                    assert int(cell) == expected
                    clues[(r, c)] = int(cell)
        assert mines <= set(hidden)
        assert count_solutions(rows, cols, clues, hidden, len(mines)) == 1


def test_score_answer():
    ds = MinesweeperDataset(MinesweeperConfig(seed=42, size=5))
    for item in ds:
        assert ds.score_answer(item["answer"], item) == 1.0
        coords = item["answer"].split("), (")
        coords = [p.strip("()") for p in coords]
        reordered = ", ".join(f"({p})" for p in reversed(coords))
        assert ds.score_answer(reordered, item) == 1.0
        assert ds.score_answer("(99,99)", item) == 0.0
        assert ds.score_answer("", item) == 0.0
        assert ds.score_answer(None, item) == 0.0


def test_fixed_board_size():
    cfg = MinesweeperConfig(seed=3, size=5, min_rows=4, max_rows=4, min_cols=4, max_cols=4, hide_safe_prob=0.3)
    ds = MinesweeperDataset(cfg)
    for item in ds:
        assert item["metadata"]["rows"] == 4 and item["metadata"]["cols"] == 4

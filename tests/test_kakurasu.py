import pytest

from reasoning_gym.coaching.base_curriculum import DefaultCurriculumContext, RangeAttributeMode
from reasoning_gym.games import KakurasuConfig, KakurasuCurriculum, KakurasuDataset


def test_kakurasu_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = KakurasuConfig(min_rows=5, max_rows=4)  # max_rows < min_rows
        config.validate()

    with pytest.raises(AssertionError):
        config = KakurasuConfig(p_ones=-0.1)  # negative probability
        config.validate()


def test_kakurasu_deterministic():
    """Test that dataset generates same puzzles with same seed"""
    config = KakurasuConfig(seed=42, size=10, min_rows=3, max_rows=9, min_cols=3, max_cols=9, p_ones=0.2)
    dataset1 = KakurasuDataset(config)
    dataset2 = KakurasuDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_kakurasu_items():
    """Test basic properties of generated items"""
    config = KakurasuConfig(seed=42, size=10, min_rows=3, max_rows=9, min_cols=3, max_cols=9, p_ones=0.3)
    dataset = KakurasuDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Verify key metadata contents
        metadata = item["metadata"]
        assert "puzzle" in metadata
        assert "solution" in metadata
        assert "row_sums" in metadata
        assert "col_sums" in metadata

        # Verify board dimensions for both puzzle and solution
        puzzle, solution = metadata["puzzle"], metadata["solution"]
        assert len(puzzle) >= config.min_rows
        assert len(puzzle) <= config.max_rows
        assert len(solution) >= config.min_rows
        assert len(solution) <= config.max_rows
        for row in puzzle:
            assert len(row) >= config.min_cols
            assert len(row) <= config.max_cols
        for row in solution:
            assert len(row) >= config.min_cols
            assert len(row) <= config.max_cols

        # Verify row and column sums
        row_sums, col_sums = metadata["row_sums"], metadata["col_sums"]
        assert len(row_sums) == len(puzzle)
        assert len(col_sums) == len(puzzle[0]) if puzzle else 0


def test_kakurasu_solution_validity():
    """Test that solutions are valid according to Kakurasu rules"""
    config = KakurasuConfig(seed=42, size=10, min_rows=3, max_rows=9, min_cols=3, max_cols=9, p_ones=0.3)
    dataset = KakurasuDataset(config)

    def is_valid_solution(solution, n_rows, n_cols, row_sums, col_sums):
        """Check if the solution is valid according to Kakurasu rules"""
        if len(solution) != n_rows or any(len(row) != n_cols for row in solution):
            return False

        # Check row sums
        for i, row in enumerate(solution):
            if sum((j + 1) for j, val in enumerate(row) if val == 1) != row_sums[i]:
                return False

        # Check column sums
        for j in range(n_cols):
            if sum((i + 1) for i, row in enumerate(solution) if row[j] == 1) != col_sums[j]:
                return False

        return True

    for i in range(len(dataset)):
        item = dataset[i]
        metadata = item["metadata"]
        solution = metadata["solution"]
        n_rows, n_cols = metadata["n_rows"], metadata["n_cols"]
        row_sums, col_sums = metadata["row_sums"], metadata["col_sums"]
        assert is_valid_solution(solution, n_rows, n_cols, row_sums, col_sums)


def test_kakurasu_puzzle_solvability():
    """Test that generated puzzles are solvable and have unique solutions"""
    config = KakurasuConfig(seed=42, size=10, min_rows=3, max_rows=9, min_cols=3, max_cols=9, p_ones=0.3)
    dataset = KakurasuDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        metadata = item["metadata"]
        n_rows, n_cols = metadata["n_rows"], metadata["n_cols"]
        row_sums, col_sums = metadata["row_sums"], metadata["col_sums"]

        # Verify puzzle has exactly one solution
        assert dataset._count_solutions(n_rows, n_cols, row_sums, col_sums) == 1


def test_kakurasu_answer_scoring():
    """Test the answer scoring mechanism"""
    config = KakurasuConfig(seed=42, size=10, min_rows=3, max_rows=9, min_cols=3, max_cols=9, p_ones=0.3)
    dataset = KakurasuDataset(config)

    for item in dataset:
        # Correct answer should score 1.0
        assert dataset.score_answer(item["answer"], item) == 1.0

        # Wrong answer should score lower
        wrong_answer = item["answer"].replace("1", "2")
        assert dataset.score_answer(wrong_answer, item) < 1.0

        # None or empty answer should score 0.0
        assert dataset.score_answer(None, item) == 0.0
        assert dataset.score_answer("", item) == 0.0


def test_futoshiki_curriculum():
    """Test the KakurasuCurriculum works as expected"""
    curriculum = KakurasuCurriculum()

    base_value = {"size": 150, "seed": 1}

    context = DefaultCurriculumContext(mode=RangeAttributeMode.UPPER_BOUND)
    base_cfg: KakurasuConfig = curriculum.generate_configuration(base_value, context=context)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_rows == 4 and base_cfg.max_rows == 4
    assert base_cfg.min_cols == 4 and base_cfg.max_cols == 4
    assert base_cfg.p_ones == 0.50

    # Test incrementing attribute levels
    curriculum.increment_attr_level("rows")
    curriculum.increment_attr_level("p_ones")
    increased_cfg = curriculum.generate_configuration(base_value, context=context)
    assert increased_cfg.min_rows == 6 and increased_cfg.max_rows == 6
    assert increased_cfg.p_ones == 0.4

    # Test incrementing again
    curriculum.increment_attr_level("cols")
    curriculum.increment_attr_level("p_ones")
    increased_cfg2 = curriculum.generate_configuration(base_value, context=context)
    assert increased_cfg2.min_cols == 6 and increased_cfg2.max_cols == 6
    assert increased_cfg2.p_ones == 0.3

    # Test incrementing to max level
    curriculum.increment_attr_level("p_ones")
    max_cfg = curriculum.generate_configuration(base_value, context=context)
    assert max_cfg.p_ones == 0.2

    # Test that we can't go beyond max level
    curriculum.increment_attr_level("p_ones")
    still_max_cfg = curriculum.generate_configuration(base_value, context=context)
    assert still_max_cfg.p_ones == 0.2

    # Test decrementing attribute levels
    curriculum.decrement_attr_level("p_ones")
    decreased_cfg = curriculum.generate_configuration(base_value, context=context)
    assert decreased_cfg.p_ones == 0.3

    # Test global level setting
    curriculum.set_global_level(0)
    global_lvl0_cfg = curriculum.generate_configuration(base_value, context=context)
    assert global_lvl0_cfg.min_rows == 4 and global_lvl0_cfg.max_rows == 4

    # Test global level increment
    curriculum.increment_global_level()
    global_lvl1_cfg = curriculum.generate_configuration(base_value, context=context)
    assert global_lvl1_cfg.min_rows == 6 and global_lvl1_cfg.max_rows == 6

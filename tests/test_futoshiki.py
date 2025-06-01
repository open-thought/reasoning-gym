import pytest

from reasoning_gym.coaching.base_curriculum import DefaultCurriculumContext, RangeAttributeMode
from reasoning_gym.games import FutoshikiConfig, FutoshikiDataset


def test_futoshiki_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = FutoshikiConfig(min_board_size=5, max_board_size=4)  # Too small
        config.validate()

    with pytest.raises(AssertionError):
        config = FutoshikiConfig(min_difficulty=2, max_difficulty=1)  # Too large
        config.validate()


def test_futoshiki_deterministic():
    """Test that dataset generates same puzzles with same seed"""
    config = FutoshikiConfig(seed=42, size=10, min_board_size=4, max_board_size=9, min_difficulty=0, max_difficulty=3)
    dataset1 = FutoshikiDataset(config)
    dataset2 = FutoshikiDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_futoshiki_items():
    """Test basic properties of generated items"""
    config = FutoshikiConfig(min_difficulty=1, max_difficulty=1, min_board_size=4, max_board_size=9, size=10, seed=42)
    dataset = FutoshikiDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Verify metadata contents
        metadata = item["metadata"]
        assert "puzzle" in metadata
        assert "solution" in metadata
        assert "constraints" in metadata

        # Verify board dimensions
        puzzle = metadata["puzzle"]
        solution = metadata["solution"]
        assert len(puzzle) >= config.min_board_size
        assert len(solution) >= config.min_board_size
        assert len(puzzle) <= config.max_board_size
        assert len(solution) <= config.max_board_size
        for row in puzzle:
            assert len(row) >= config.min_board_size
            assert len(row) <= config.max_board_size
        for row in solution:
            assert len(row) >= config.min_board_size
            assert len(row) <= config.max_board_size
        # Verify constraints format
        constraints = metadata["constraints"]
        for r1, c1, r2, c2, rel in constraints:
            assert 0 <= r1 < config.max_board_size
            assert 0 <= c1 < config.max_board_size
            assert 0 <= r2 < config.max_board_size
            assert 0 <= c2 < config.max_board_size
            assert rel in ("<", ">")


def test_futoshiki_solution_validity():
    """Test that solutions are valid according to Futoshiki rules"""
    config = FutoshikiConfig(min_board_size=4, max_board_size=4, min_difficulty=1, max_difficulty=1, size=10, seed=42)
    dataset = FutoshikiDataset(config)

    def is_valid_solution(solution, board_size, constraints):
        # Check rows
        for row in solution:
            if sorted(row) != list(range(1, board_size + 1)):
                return False

        # Check columns
        for col in range(board_size):
            column = [solution[row][col] for row in range(board_size)]
            if sorted(column) != list(range(1, board_size + 1)):
                return False

        # Check constraints
        for ((r1, c1), (r2, c2)), rel in constraints.items():
            v1, v2 = solution[r1][c1], solution[r2][c2]
            if rel == "<" and not (v1 < v2):
                return False
            if rel == ">" and not (v1 > v2):
                return False

        return True

    for i in range(len(dataset)):
        item = dataset[i]
        metadata = item["metadata"]
        solution = metadata["solution"]
        constraints_meta = metadata["constraints"]

        constraints = {((r1, c1), (r2, c2)): rel for (r1, c1, r2, c2, rel) in constraints_meta}

        assert is_valid_solution(solution, config.min_board_size, constraints)


def test_futoshiki_puzzle_solvability():
    """Test that generated puzzles are solvable and have unique solutions"""
    config = FutoshikiConfig(min_board_size=4, max_board_size=4, min_difficulty=1, max_difficulty=1, size=5, seed=42)
    dataset = FutoshikiDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        metadata = item["metadata"]
        puzzle = metadata["puzzle"]
        constraints_meta = metadata["constraints"]

        constraints = {((r1, c1), (r2, c2)): rel for (r1, c1, r2, c2, rel) in constraints_meta}

        # Verify puzzle has exactly one solution
        assert dataset.count_solutions(puzzle, constraints, limit=2) == 1


def test_futoshiki_difficulty_levels():
    """Test that different difficulty levels affect puzzle complexity"""
    size = 5
    board_size = 4
    seeds = [42, 43, 44]  # Test multiple seeds for robustness

    def count_clues(puzzle):
        return sum(cell != 0 for row in puzzle for cell in row)

    def count_constraints(constraints):
        return len(constraints)

    for seed in seeds:
        clues_by_difficulty = []
        constraints_by_difficulty = []

        for difficulty in range(4):  # 0 to 3
            config = FutoshikiConfig(
                min_board_size=board_size,
                max_board_size=board_size,
                min_difficulty=difficulty,
                max_difficulty=difficulty,
                size=size,
                seed=seed,
            )
            dataset = FutoshikiDataset(config)

            avg_clues = sum(count_clues(item["metadata"]["puzzle"]) for item in dataset) / size
            avg_constraints = sum(count_constraints(item["metadata"]["constraints"]) for item in dataset) / size

            clues_by_difficulty.append(avg_clues)
            constraints_by_difficulty.append(avg_constraints)

        # Higher difficulty should generally mean fewer clues and/or more constraints
        assert all(clues_by_difficulty[i] >= clues_by_difficulty[i + 1] for i in range(len(clues_by_difficulty) - 1))
        assert all(
            constraints_by_difficulty[i] <= constraints_by_difficulty[i + 1]
            for i in range(len(constraints_by_difficulty) - 1)
        )


def test_futoshiki_answer_scoring():
    """Test the answer scoring mechanism"""
    config = FutoshikiConfig(min_board_size=4, max_board_size=4, min_difficulty=0, max_difficulty=0, size=5, seed=42)
    dataset = FutoshikiDataset(config)

    for item in dataset:
        # Correct answer should score 1.0
        assert dataset.score_answer(item["answer"], item) == 1.0

        # Wrong answer should score lower
        wrong_answer = item["answer"].replace("1", "2")
        assert dataset.score_answer(wrong_answer, item) < 1.0

        # None or empty answer should score 0.0
        assert dataset.score_answer(None, item) == 0.0
        assert dataset.score_answer("", item) == 0.0

        answer = item["answer"]
        white_space_mismatch = answer.replace("   ", " ")
        assert dataset.score_answer(white_space_mismatch, item) == 0.9

        anwser_with_additional_text = "This is an anwser " + answer + "\nwith surrounding text."
        assert 0 < dataset.score_answer(anwser_with_additional_text, item) < 0.9

        partially_correct = anwser_with_additional_text.replace("1", "2")
        assert dataset.score_answer(partially_correct, item) > 0.1

        bad_answer = "\n".join(anwser_with_additional_text.split("\n")[::-1])
        assert dataset.score_answer(bad_answer, item) < 0.1


def test_futoshiki_curriculum():
    """Test the FutoshikiCurriculum works as expected"""
    from reasoning_gym.games.futoshiki import FutoshikiCurriculum

    curriculum = FutoshikiCurriculum()

    base_value = {"size": 150, "seed": 1}

    context = DefaultCurriculumContext(mode=RangeAttributeMode.UPPER_BOUND)
    base_cfg: FutoshikiConfig = curriculum.generate_configuration(base_value, context=context)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_board_size == 4 and base_cfg.max_board_size == 4
    assert base_cfg.min_difficulty == 0 and base_cfg.max_difficulty == 0

    # Test incrementing attribute levels
    curriculum.increment_attr_level("board_size")
    curriculum.increment_attr_level("difficulty")
    increased_cfg = curriculum.generate_configuration(base_value, context=context)
    assert increased_cfg.min_board_size == 6 and increased_cfg.max_board_size == 6
    assert increased_cfg.min_difficulty == 1 and increased_cfg.max_difficulty == 1

    # Test incrementing again
    curriculum.increment_attr_level("board_size")
    curriculum.increment_attr_level("difficulty")
    increased_cfg2 = curriculum.generate_configuration(base_value, context=context)
    assert increased_cfg2.min_board_size == 7 and increased_cfg2.max_board_size == 7
    assert increased_cfg2.min_difficulty == 2 and increased_cfg2.max_difficulty == 2

    # Test incrementing to max levels
    curriculum.increment_attr_level("board_size")
    curriculum.increment_attr_level("difficulty")
    max_cfg = curriculum.generate_configuration(base_value, context=context)
    assert max_cfg.min_board_size == 9 and max_cfg.max_board_size == 9
    assert max_cfg.min_difficulty == 3 and max_cfg.max_difficulty == 3

    # Test that we can't go beyond max levels
    assert not curriculum.increment_attr_level("board_size")
    assert not curriculum.increment_attr_level("difficulty")
    still_max_cfg = curriculum.generate_configuration(base_value, context=context)
    assert still_max_cfg.min_board_size == 9 and still_max_cfg.max_board_size == 9
    assert still_max_cfg.min_difficulty == 3 and still_max_cfg.max_difficulty == 3

    # Test decrementing attribute levels
    curriculum.decrement_attr_level("board_size")
    curriculum.decrement_attr_level("difficulty")
    decreased_cfg = curriculum.generate_configuration(base_value, context=context)
    assert decreased_cfg.min_board_size == 7 and decreased_cfg.max_board_size == 7
    assert decreased_cfg.min_difficulty == 2 and decreased_cfg.max_difficulty == 2

    # Test global level setting
    curriculum.set_global_level(0)
    global_lvl0_cfg = curriculum.generate_configuration(base_value, context=context)
    assert global_lvl0_cfg.min_board_size == 4 and global_lvl0_cfg.max_board_size == 4
    assert global_lvl0_cfg.min_difficulty == 0 and global_lvl0_cfg.max_difficulty == 0

    # Test global level increment
    curriculum.increment_global_level()
    global_lvl1_cfg = curriculum.generate_configuration(base_value, context=context)
    assert global_lvl1_cfg.min_board_size == 6 and global_lvl1_cfg.max_board_size == 6
    assert global_lvl1_cfg.min_difficulty == 1 and global_lvl1_cfg.max_difficulty == 1

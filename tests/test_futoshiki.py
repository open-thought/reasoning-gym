import pytest

from reasoning_gym.games import FutoshikiConfig, FutoshikiDataset


def test_futoshiki_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = FutoshikiConfig(board_size=3)  # Too small
        config.validate()

    with pytest.raises(AssertionError):
        config = FutoshikiConfig(board_size=10)  # Too large
        config.validate()

    with pytest.raises(AssertionError):
        config = FutoshikiConfig(difficulty=-1)  # Invalid difficulty
        config.validate()

    with pytest.raises(AssertionError):
        config = FutoshikiConfig(difficulty=4)  # Invalid difficulty
        config.validate()


def test_futoshiki_deterministic():
    """Test that dataset generates same puzzles with same seed"""
    config = FutoshikiConfig(seed=42, size=10)
    dataset1 = FutoshikiDataset(config)
    dataset2 = FutoshikiDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_futoshiki_items():
    """Test basic properties of generated items"""
    config = FutoshikiConfig(board_size=4, difficulty=1, size=10, seed=42)
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
        assert "board_size" in metadata
        assert "difficulty" in metadata

        # Verify board dimensions
        puzzle = metadata["puzzle"]
        solution = metadata["solution"]
        assert len(puzzle) == config.board_size
        assert len(solution) == config.board_size
        for row in puzzle:
            assert len(row) == config.board_size
        for row in solution:
            assert len(row) == config.board_size

        # Verify constraints format
        constraints = metadata["constraints"]
        for ((r1, c1), (r2, c2)), rel in constraints.items():
            assert 0 <= r1 < config.board_size
            assert 0 <= c1 < config.board_size
            assert 0 <= r2 < config.board_size
            assert 0 <= c2 < config.board_size
            assert rel in ("<", ">")


def test_futoshiki_solution_validity():
    """Test that solutions are valid according to Futoshiki rules"""
    config = FutoshikiConfig(board_size=4, difficulty=1, size=10, seed=42)
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
        constraints = metadata["constraints"]

        assert is_valid_solution(solution, config.board_size, constraints)


def test_futoshiki_puzzle_solvability():
    """Test that generated puzzles are solvable and have unique solutions"""
    config = FutoshikiConfig(board_size=4, difficulty=1, size=5, seed=42)
    dataset = FutoshikiDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        metadata = item["metadata"]
        puzzle = metadata["puzzle"]
        constraints = metadata["constraints"]

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
            config = FutoshikiConfig(board_size=board_size, difficulty=difficulty, size=size, seed=seed)
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
    config = FutoshikiConfig(board_size=4, difficulty=0, size=5, seed=42)
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

import pytest

from reasoning_gym.logic.knights_knaves import KnightsKnavesConfig, KnightsKnavesDataset


def test_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = KnightsKnavesConfig(n_people=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = KnightsKnavesConfig(depth_constraint=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = KnightsKnavesConfig(width_constraint=0)
        config.validate()


def test_deterministic():
    """Test that dataset generates same items with same seed"""
    config = KnightsKnavesConfig(seed=42, size=10)
    dataset1 = KnightsKnavesDataset(config)
    dataset2 = KnightsKnavesDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_items():
    """Test basic properties of generated items"""
    config = KnightsKnavesConfig(
        n_people=2,
        depth_constraint=2,
        width_constraint=2,
        size=100,
        seed=42,
    )
    dataset = KnightsKnavesDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item


def test_solution():
    config = KnightsKnavesConfig(
        n_people=2,
        depth_constraint=2,
        width_constraint=2,
        size=100,
        seed=42,
    )
    dataset = KnightsKnavesDataset(config)

    problem = dataset[0]
    solutions = KnightsKnavesDataset.find_solution(problem["metadata"]["statements"])
    assert len(solutions) == 1, "Should have exactly one solution"
    assert solutions[0] == problem["metadata"]["solution"], "find_solution should match metadata solution"


def test_specific_problem():
    """Test a specific problem from the dataset"""
    test_statements = (
        ("or", ("telling-truth", 3), ("telling-truth", 2)),
        ("not", ("telling-truth", 2)),
        ("->", ("lying", 0), ("telling-truth", 3)),
        ("->", ("lying", 1), ("lying", 4)),
        ("not", ("lying", 0)),
    )
    test_solutions = KnightsKnavesDataset.find_solution(test_statements)
    assert test_solutions == [
        (True, False, True, False, True)
    ], "Solution should be [(True, False, True, False, True)] for test example"


def test_score_answer():
    config = KnightsKnavesConfig(
        n_people=2,
        depth_constraint=2,
        width_constraint=2,
        size=100,
        seed=42,
    )
    dataset = KnightsKnavesDataset(config)
    problem = dataset[0]
    correct_answer = problem["answer"]  ## 'Zoey is a fool, and Riley is a sage.'
    print(correct_answer)
    half_answer = "Zoey is a fool and Riley is a fool."
    wrong_answer = "Zoey sage Riley fool"
    modified_answer = "(Zoey, fool), (Riley, sage)"
    flipped_answer = "(Riley,sage), (Zoey,fool)"

    assert dataset.score_answer(correct_answer, problem) == 1.0
    assert abs(dataset.score_answer(half_answer, problem) - 0.65) < 1e-10
    assert dataset.score_answer(modified_answer, problem) == 1.0
    assert dataset.score_answer(wrong_answer, problem) == 0.01
    print("flipped")
    assert dataset.score_answer(flipped_answer, problem) == 1.0


def test_people_count():
    """Test that different parameters generate different problems"""
    config_more_people = KnightsKnavesConfig(
        n_people=4,
        depth_constraint=2,
        width_constraint=2,
        size=100,
        seed=42,
    )
    dataset_more_people = KnightsKnavesDataset(config_more_people)
    item = dataset_more_people[0]
    assert len(item["metadata"]["names"]) == 4


def test_zero_multiple_solutions():
    no_solution_statements = (("telling-truth", 1), ("lying", 0))
    solutions = KnightsKnavesDataset.find_solution(no_solution_statements)
    assert len(solutions) == 0, "Should have no solutions for contradictory statements"

    multiple_solutions_statements = (("telling-truth", 1), ("telling-truth", 0))
    solutions = KnightsKnavesDataset.find_solution(multiple_solutions_statements)
    assert len(solutions) == 2, "Should have two solutions for consistent statements"


def test_invalid_statements():
    """Test handling of invalid statements"""
    with pytest.raises(KeyError):
        KnightsKnavesDataset.test_satisfiability(("invalid-operator", 0), (True,))


def test_normalize_answer():
    """Test normalization of answer strings"""
    answer1 = "Zoey is a fool, and Riley is a sage."
    answer2 = "Zoey fool, Riley sage"
    answer3 = "(Zoey, fool), (Riley, sage)"
    answer4 = "(Riley, sage), (Zoey, fool)"
    normalized1 = KnightsKnavesDataset._normalize_answer(answer1)
    assert normalized1 == {("zoey", "fool"), ("riley", "sage")}
    normalized2 = KnightsKnavesDataset._normalize_answer(answer2)
    assert normalized2 == {("zoey", "fool"), ("riley", "sage")}
    normalized3 = KnightsKnavesDataset._normalize_answer(answer3)
    assert normalized3 == {("zoey", "fool"), ("riley", "sage")}
    normalized4 = KnightsKnavesDataset._normalize_answer(answer4)
    assert normalized4 == {("zoey", "fool"), ("riley", "sage")}


def test_satisfiability():
    """Test the test_satisfiability method with different logical operators"""
    # Test basic operators
    assert KnightsKnavesDataset.test_satisfiability(("telling-truth", 0), (True,))
    assert not KnightsKnavesDataset.test_satisfiability(("telling-truth", 0), (False,))
    assert not KnightsKnavesDataset.test_satisfiability(("lying", 0), (True,))
    assert KnightsKnavesDataset.test_satisfiability(("lying", 0), (False,))

    # Test NOT
    assert not KnightsKnavesDataset.test_satisfiability(("not", ("telling-truth", 0)), (True,))
    assert KnightsKnavesDataset.test_satisfiability(("not", ("telling-truth", 0)), (False,))

    # Test AND
    assert KnightsKnavesDataset.test_satisfiability(("and", ("telling-truth", 0), ("telling-truth", 1)), (True, True))
    assert not KnightsKnavesDataset.test_satisfiability(
        ("and", ("telling-truth", 0), ("telling-truth", 1)), (True, False)
    )

    # Test OR
    assert KnightsKnavesDataset.test_satisfiability(("or", ("telling-truth", 0), ("telling-truth", 1)), (True, False))
    assert not KnightsKnavesDataset.test_satisfiability(
        ("or", ("telling-truth", 0), ("telling-truth", 1)), (False, False)
    )

    # Test implication
    assert KnightsKnavesDataset.test_satisfiability(("->", ("telling-truth", 0), ("telling-truth", 1)), (False, False))
    assert not KnightsKnavesDataset.test_satisfiability(
        ("->", ("telling-truth", 0), ("telling-truth", 1)), (True, False)
    )

    # Test bi-implication
    assert KnightsKnavesDataset.test_satisfiability(("<=>", ("telling-truth", 0), ("telling-truth", 1)), (True, True))
    assert KnightsKnavesDataset.test_satisfiability(("<=>", ("telling-truth", 0), ("telling-truth", 1)), (False, False))
    assert not KnightsKnavesDataset.test_satisfiability(
        ("<=>", ("telling-truth", 0), ("telling-truth", 1)), (True, False)
    )

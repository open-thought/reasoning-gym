"""Tests for String Splitting questions generation"""

import pytest

from reasoning_gym.algorithmic.string_splitting import StringSplittingConfig, StringSplittingDataset

QUESTION_TEMPLATE = """There is a dismantling engineer who has old machines A, B, and C.
He discovered that he can obtain a batch of new parts X, Y, Z through the following rules:
1. One unit of machine A can be dismanteled into two units of part X and one unit of part Y.
2. Two units of machine B can be dismanteled into one unit of part X.
3. Two units of machine C can be dismanteled into one unit of part Y.
4. One unit of machine B and one unit of machine C can be combined into one unit of machine A.
5. One unit of part X and one unit of part Y can be combined into one unit of part Z.

Given a certain number of initial machines, your job is to continuously cycle through the rules 1-5 above, exausting one rule at a time, until no more rules can be applied, or until a state (counts of each machine and part type) is repeated.
After you make use of a rule, you should update the counts of each machine and part type accordingly, and then restart the process from rule 1.

The output should be the count of each machine and part type after the rules have been exhaustively applied in the following order: A B C X Y Z.
For example 1 0 1 5 4 3 means that you have 1 machine A, 0 machine B, 1 machine C, 5 part X, 4 part Y, and 3 part Z.

Example:
- Input: You have 2 machines A, 0 machines B, and 1 machine C.
- Output: 0 0 1 2 0 2
- Explanation
    0. Initial state: 2 0 1 0 0 0
    1. We can apply rule 1 and trade 1 machine A for 2 part X and 1 part Y: 1 0 1 2 1 0
    2. Starting over, we can apply rule 1 again: 0 0 1 4 2 0
    3. In the next iteration, we can apply rule 5 and trade 1 part X and 1 part Y for 1 part Z: 0 0 1 3 1 1
    4. In the next iteration, we can apply rule 5 again: 0 0 1 2 0 2
    5. We can't apply any more rules, so the final answer is 0 0 1 2 0 2

Now, you have {A_machine} machine A, {B_machine} machine B, and {C_machine} machine C. Provide the count of each machine and part type after applying the above rules.
"""


def test_string_splitting_config_validation():
    """Test that invalid configs raise appropriate errors"""

    with pytest.raises(AssertionError):
        config = StringSplittingConfig(min_initial_machines=-1)  # negative not allowed
        config.validate()

    with pytest.raises(AssertionError):
        config = StringSplittingConfig(min_initial_machines=3, max_initial_machines=2)  # min > max
        config.validate()


def test_string_splitting_dataset_deterministic():
    """Test that dataset generates same items with same seed"""
    config = StringSplittingConfig(seed=42, size=10)
    dataset1 = StringSplittingDataset(config)
    dataset2 = StringSplittingDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_string_splitting_dataset_items():
    """Test basic properties of generated items"""
    config = StringSplittingConfig(min_initial_machines=1, max_initial_machines=5, size=10, seed=42)
    dataset = StringSplittingDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        # Check item structure
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata
        assert "states" in item["metadata"]
        assert "solution" in item["metadata"]

        states = item["metadata"]["states"]
        solution = item["metadata"]["solution"]

        # Verify dimensions
        assert len(states) > 0
        assert states[-1] == solution
        for i in range(3):
            assert 1 <= states[0][i] <= 5
        for i in range(3, 6):
            assert states[0][i] == 0


def test_string_splitting_dataset_iteration():
    """Test that iteration respects dataset size"""
    config = StringSplittingConfig(size=5, seed=42)
    dataset = StringSplittingDataset(config)

    items = list(dataset)
    assert len(items) == config.size

    # Test multiple iterations yield same items
    assert items == list(dataset)


def test_string_splitting_answer():
    """Test the answer calculation"""
    config = StringSplittingConfig(seed=42)
    dataset = StringSplittingDataset(config)

    # Empty input
    counts = [0, 0, 0, 0, 0, 0]
    assert dataset._apply_rule(counts) == [0, 0, 0, 0, 0, 0]

    # Rule 1: 1A -> 2X 1Y
    counts = [1, 0, 0, 0, 0, 0]
    assert dataset._apply_rule(counts) == [0, 0, 0, 2, 1, 0]

    # Rule 2: 2B -> 1X
    counts = [0, 2, 0, 0, 0, 0]
    assert dataset._apply_rule(counts) == [0, 0, 0, 1, 0, 0]

    # Rule 3: 2C -> 1Y
    counts = [0, 0, 2, 0, 0, 0]
    assert dataset._apply_rule(counts) == [0, 0, 0, 0, 1, 0]

    # Rule 4: B + C -> A
    counts = [0, 1, 1, 0, 0, 0]
    assert dataset._apply_rule(counts) == [1, 0, 0, 0, 0, 0]

    # Rule 5: X + Y -> Z
    counts = [0, 0, 0, 1, 1, 0]
    assert dataset._apply_rule(counts) == [0, 0, 0, 0, 0, 1]

    # 1-shot example used in the prompt
    A_machine, B_machine, C_machine = 2, 0, 1
    assert dataset._get_answer(A_machine, B_machine, C_machine) == [
        [2, 0, 1, 0, 0, 0],
        [1, 0, 1, 2, 1, 0],
        [0, 0, 1, 4, 2, 0],
        [0, 0, 1, 3, 1, 1],
        [0, 0, 1, 2, 0, 2],
    ]

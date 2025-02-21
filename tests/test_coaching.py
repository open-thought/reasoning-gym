import json
import math
from collections import OrderedDict

import pytest

from reasoning_gym.arithmetic.chain_sum import ChainSumConfig, ChainSumDataset
from reasoning_gym.arithmetic.leg_counting import LegCountingConfig
from reasoning_gym.coaching import Coach, GroupedScores
from reasoning_gym.composite import CompositeConfig, CompositeDataset, DatasetSpec


def test_coach_with_chain_sum():
    # Create a small ChainSum dataset
    config = ChainSumConfig(min_terms=2, max_terms=3, min_digits=1, max_digits=2, size=10, seed=42)
    dataset = ChainSumDataset(config)
    coach = Coach(dataset)

    # Simulate an agent working on tasks
    for i in range(5):
        item = coach[i]

        # Simulate some correct and incorrect answers
        if i % 2 == 0:
            # Correct answer
            score = coach.score_answer(
                answer=item["answer"],
                entry=item,
                conversation=[
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["answer"]},
                ],
            )
            assert score == 1.0
        else:
            # Incorrect answer (None)
            score = coach.score_answer(
                answer=None,
                entry=item,
                conversation=[
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": "I don't know"},
                ],
            )
            assert score == 0.0

    # Test score aggregation
    aggregated = coach.score_board.aggregate()

    # Verify we have scores grouped by difficulty parameters
    assert len(aggregated.scores) > 0

    # Each key should be a tuple of tuples containing difficulty parameters
    for key in aggregated.scores:
        assert isinstance(key, tuple)
        # Each inner tuple should be (param_name, value)
        for param in key:
            assert isinstance(param, tuple)
            assert param[0] in ("num_terms", "num_digits")
            assert isinstance(param[1], int)

    # Test aggregation with last_n
    last_3 = coach.score_board.aggregate(last_n=3)
    assert len(last_3.scores) > 0

    # Verify total scores count
    assert last_3.total_scores == 3

    # Verify conversation tracking
    assert len(coach.score_board.conversations) == 5
    for conv in coach.score_board.conversations:
        assert len(conv) == 2  # user question and assistant response
        assert conv[0]["role"] == "user"
        assert conv[1]["role"] == "assistant"

    # Test stats calculation
    stats = aggregated.stats()

    for key, values in stats.scores.items():
        assert isinstance(values, tuple)
        assert len(values) == 5  # (count, mean, std, min, max)
        assert isinstance(values[0], int)  # count should be int
        assert all(isinstance(v, float) for v in values[1:])  # stats should be floats

    # Test stats with empty scores
    empty_stats = GroupedScores(scores=OrderedDict(), total_scores=0).stats()
    assert len(empty_stats.scores) == 0

    # Test stats with ignore_empty=False
    empty_group = OrderedDict({(("test", 1),): []})
    non_ignoring_stats = GroupedScores(scores=empty_group, total_scores=0).stats(ignore_empty=False)
    assert len(non_ignoring_stats.scores) == 1
    stats_tuple = next(iter(non_ignoring_stats.scores.values()))
    assert stats_tuple[0] == 0  # count should be 0 for empty list
    assert all(math.isnan(v) for v in stats_tuple[1:])  # stats should be NaN

    print(aggregated)
    print(stats)

    # Test clear functionality
    coach.score_board.clear()
    assert len(coach.score_board.scores) == 0
    assert len(coach.score_board.metadata) == 0
    assert len(coach.score_board.conversations) == 0
    assert len(coach.score_board.aggregate().scores) == 0


def test_coach_with_composite():
    # Create configs for both datasets
    chain_sum_config = ChainSumConfig(min_terms=2, max_terms=3, min_digits=1, max_digits=2, size=10)
    leg_counting_config = LegCountingConfig(min_animals=2, max_animals=3, size=10)

    # Create composite config
    composite_config = CompositeConfig(
        size=20,
        seed=42,
        datasets=[
            DatasetSpec(name="chain_sum", weight=1.0, config=chain_sum_config.__dict__),
            DatasetSpec(name="leg_counting", weight=1.0, config=leg_counting_config.__dict__),
        ],
    )

    # Create composite dataset and coach
    dataset = CompositeDataset(composite_config)
    coach = Coach(dataset)

    # Score some answers
    for i in range(5):
        item = coach[i]
        # Correct answers for even indices
        score = coach.score_answer(
            answer=item["answer"] if i % 2 == 0 else None,
            entry=item,
            conversation=[
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"] if i % 2 == 0 else "I don't know"},
            ],
        )
        assert score in (0.0, 1.0)

    # Test aggregation
    aggregated = coach.score_board.aggregate()
    assert len(aggregated.scores) > 0

    # Verify source dataset info is first in keys
    for key in aggregated.scores:
        assert key[0][0] == "source"  # First tuple should be ("source", dataset_name)
        assert key[1][0] == "idx"  # Second tuple should be ("idx", index)

    # Test stats
    stats = aggregated.stats()
    for key, values in stats.scores.items():
        assert isinstance(values, tuple)
        assert len(values) == 5  # (count, mean, std, min, max)
        assert isinstance(values[0], int)
        assert all(isinstance(v, float) for v in values[1:])

    print("\nComposite Dataset Stats:")
    print(stats)

    # Test config update
    coach.dataset.update_dataset_config("chain_sum", {"min_terms": 4, "max_terms": 5})

    # Verify the config was updated
    chain_sum_dataset = coach.dataset.datasets["chain_sum"]
    assert chain_sum_dataset.config.min_terms == 4
    assert chain_sum_dataset.config.max_terms == 5

    # Score some more items to verify new config is in effect
    for i in range(3):
        item = coach[i + 5]  # Use different indices
        if "chain_sum" in item["metadata"]["source_dataset"]:
            metadata = item["metadata"]
            assert metadata["difficulty"]["num_terms"] >= 4


def test_grouped_scores_str():
    # Test raw scores string representation
    scores = OrderedDict()
    scores[(("num_terms", 2), ("num_digits", 1))] = [1.0, 0.0, 1.0]
    scores[(("num_terms", 3), ("num_digits", 2))] = [0.5, 0.5]
    grouped = GroupedScores(scores=scores, total_scores=5)

    report = str(grouped)
    assert "Total scores: 5" in report
    assert "(num_terms=2, num_digits=1): n=3" in report
    assert "(num_terms=3, num_digits=2): n=2" in report
    assert "Values: 1.00, 0.00, 1.00" in report
    assert "Values: 0.50, 0.50" in report

    # Test stats string representation
    stats = grouped.stats()
    stats_report = str(stats)
    assert "μ=" in stats_report
    assert "σ=" in stats_report
    assert "min=" in stats_report
    assert "max=" in stats_report

    # Test empty scores
    empty = GroupedScores(scores=OrderedDict(), total_scores=0)
    assert str(empty) == "No scores recorded"


def test_coach_score_logging(tmp_path):
    # Create a log file in the temporary directory
    log_file = tmp_path / "scores.jsonl"

    # Create dataset and coach with logging
    config = ChainSumConfig(min_terms=2, max_terms=3, min_digits=1, max_digits=2, size=10, seed=42)
    dataset = ChainSumDataset(config)
    coach = Coach(dataset, score_log=log_file)

    # Score a few answers
    for i in range(3):
        item = coach[i]
        coach.score_answer(
            answer=item["answer"] if i % 2 == 0 else None,
            entry=item,
            conversation=[
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"] if i % 2 == 0 else "I don't know"},
            ],
        )

    # Verify log file contents
    assert log_file.exists()

    # Read and parse log entries
    log_entries = [json.loads(line) for line in log_file.open()]
    assert len(log_entries) == 3

    # Verify log entry structure
    for i, entry in enumerate(log_entries):
        assert "score" in entry
        assert "entry" in entry
        assert "metadata" in entry["entry"]
        assert "conversation" in entry
        assert entry["score"] == (1.0 if i % 2 == 0 else 0.0)
        assert len(entry["conversation"]) == 2

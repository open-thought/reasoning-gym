import math
from collections import OrderedDict

import pytest

from reasoning_gym.arithmetic.chain_sum import ChainSumConfig, ChainSumDataset
from reasoning_gym.arithmetic.leg_counting import LegCountingConfig
from reasoning_gym.coaching import (
    CurriculumAttributeConfig,
    CurriculumExperiment,
    CurriculumExperimentConfig,
    GroupedScores,
)
from reasoning_gym.coaching.base_curriculum import DefaultCurriculumContext, RangeAttributeMode
from reasoning_gym.composite import CompositeConfig, CompositeDataset, DatasetSpec


def test_score_aggregation():
    config = CurriculumExperimentConfig(
        curricula={"leg_counting": CurriculumAttributeConfig(attribute_levels={"num_animals": 2}, weight=1.0)}
    )

    # Create experiment
    experiment = CurriculumExperiment(
        name="test_experiment",
        config=config,
        context=DefaultCurriculumContext(mode=RangeAttributeMode.INCLUSIVE),
        size=10,
        seed=42,
    )

    # Simulate an agent working on tasks
    for i in range(5):
        item = experiment.get_dataset_entry(i)

        # Simulate some correct and incorrect answers
        if i % 2 == 0:
            # Correct answer
            score = experiment.score_answer_with_id(
                answer=item["answer"],
                entry_id=item["metadata"]["entry_id"],
                conversation=[
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["answer"]},
                ],
            )
            assert score == 1.0
        else:
            # Incorrect answer (None)
            score = experiment.score_answer_with_id(
                answer=None,
                entry_id=item["metadata"]["entry_id"],
                conversation=[
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["answer"]},
                ],
            )
            assert score == 0.0

    # Test score aggregation
    aggregated = experiment.score_board.aggregate()

    # Verify we have scores grouped by difficulty parameters
    assert len(aggregated.scores) > 0

    # Each key should be a tuple of tuples containing difficulty parameters
    for key in aggregated.scores:
        assert isinstance(key, tuple)
        # Each inner tuple should be (param_name, value) or (param_name, (min_value, max_value))
        for param in key:
            assert isinstance(param, tuple)
            assert param[0] in ("source", "idx", "num_animals", "num_instances")

    # Test aggregation with last_n
    last_3 = experiment.score_board.aggregate(last_n=3)
    assert len(last_3.scores) > 0

    # Verify total scores count
    assert last_3.total_scores == 3

    # Verify conversation tracking
    assert len(experiment.score_board.conversations) == 5
    for conv in experiment.score_board.conversations:
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

    # Test clear functionality
    experiment.score_board.clear()
    assert len(experiment.score_board.scores) == 0
    assert len(experiment.score_board.metadata) == 0
    assert len(experiment.score_board.conversations) == 0
    assert len(experiment.score_board.aggregate().scores) == 0


def test_experiment_with_composite():
    # Create configs for both datasets
    config = CurriculumExperimentConfig(
        curricula={
            "chain_sum": CurriculumAttributeConfig(attribute_levels={"num_terms": 2}, weight=1.0),
            "leg_counting": CurriculumAttributeConfig(attribute_levels={"num_animals": 2}, weight=1.0),
        }
    )
    # Create experiment
    experiment = CurriculumExperiment(
        name="test_experiment",
        config=config,
        context=DefaultCurriculumContext(mode=RangeAttributeMode.INCLUSIVE),
        size=10,
        seed=42,
    )

    # Score some answers
    for i in range(5):
        item = experiment.get_dataset_entry(i)
        # Correct answers for even indices
        score = experiment.score_answer_with_id(
            answer=item["answer"] if i % 2 == 0 else None,
            entry_id=item["metadata"]["entry_id"],
            conversation=[
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"] if i % 2 == 0 else "I don't know"},
            ],
        )
        assert score in (0.0, 1.0)

    # Test aggregation
    aggregated = experiment.score_board.aggregate()
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

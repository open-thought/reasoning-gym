import pytest

from reasoning_gym.arithmetic.chain_sum import ChainSum, ChainSumConfig
from reasoning_gym.coaching import Coach


def test_coach_with_chain_sum():
    # Create a small ChainSum dataset
    config = ChainSumConfig(
        min_terms=2,
        max_terms=3,
        min_digits=1,
        max_digits=2,
        size=10,
        seed=42
    )
    dataset = ChainSum(config)
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
                    {"role": "assistant", "content": item["answer"]}
                ]
            )
            assert score == 1.0
        else:
            # Incorrect answer (None)
            score = coach.score_answer(
                answer=None,
                entry=item,
                conversation=[
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": "I don't know"}
                ]
            )
            assert score == 0.0
    
    # Test score aggregation
    aggregated = coach.score_board.aggregate()
    
    # Verify we have scores grouped by difficulty parameters
    assert len(aggregated) > 0
    
    # Each key should be a tuple of tuples containing difficulty parameters
    for key in aggregated:
        assert isinstance(key, tuple)
        # Each inner tuple should be (param_name, value)
        for param in key:
            assert isinstance(param, tuple)
            assert param[0] in ("num_terms", "num_digits")
            assert isinstance(param[1], int)
    
    # Test aggregation with last_n
    last_3 = coach.score_board.aggregate(last_n=3)
    assert len(last_3) > 0
    
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
    assert stats.total_scores == -1  # Indicates stats object
    
    for key, values in stats.scores.items():
        assert len(values) == 4  # [mean, std, min, max]
        assert all(isinstance(v, float) for v in values)
        
    # Test stats with empty scores
    empty_stats = GroupedScores(scores=OrderedDict(), total_scores=0).stats()
    assert len(empty_stats.scores) == 0
    
    # Test stats with ignore_empty=False
    empty_group = OrderedDict({(("test", 1),): []})
    non_ignoring_stats = GroupedScores(scores=empty_group, total_scores=0).stats(ignore_empty=False)
    assert len(non_ignoring_stats.scores) == 1
    assert all(math.isnan(v) for v in next(iter(non_ignoring_stats.scores.values())))

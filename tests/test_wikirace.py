import pytest

from reasoning_gym.games.wikirace import WikiraceConfig, WikiraceCurriculum, WikiraceDataset


def test_wikirace_game_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = WikiraceConfig(min_distance=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = WikiraceConfig(min_distance=3, max_distance=2)
        config.validate()

    with pytest.raises(AssertionError):
        config = WikiraceConfig(max_tries=-2)
        config.validate()


def test_wikirace_game_deterministic():
    """Test that dataset generates same items with same seed"""
    config1 = WikiraceConfig(seed=42, size=2)
    dataset1 = WikiraceDataset(config1)
    config2 = WikiraceConfig(seed=42, size=2)
    dataset2 = WikiraceDataset(config2)

    for i in range(len(dataset1)):
        assert dataset1[i] == dataset2[i]


def test_wikirace_game_items():
    """Test basic properties of generated items"""
    config = WikiraceConfig(
        seed=42,
        size=2,
    )
    dataset = WikiraceDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata contains required fields
        assert "source" in item["metadata"]
        assert "links" in item["metadata"]
        assert "target" in item["metadata"]
        assert "current" in item["metadata"]
        assert "distance" in item["metadata"]

        # Verify number of source numbers is within config range
        assert config.min_distance <= item["metadata"]["distance"] <= config.max_distance

        # A non-int answer fails
        assert dataset.score_answer(answer="nope", entry=item) == 0.01

        # A negative answer fails
        assert dataset.score_answer(answer="-1", entry=item) == 0.01

        # An out of bond answer fails
        assert dataset.score_answer(answer=str(len(item["metadata"]["links"])), entry=item) == 0.01

        # A parsable answer gives at least 0.1
        assert dataset.score_answer(answer="0", entry=item) >= 0.1

        # The expected answer gives 1.0
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0


def test_wikirace_game_single():
    """Test a known item"""
    config = WikiraceConfig(
        seed=42,
        size=1,
    )
    dataset = WikiraceDataset(config)
    item = dataset[0]

    # If those asserts fails, it probably just means you changed the generation algorithm, which is fine
    # you'll have have to update this test
    assert item["metadata"]["source"] == "Vadim Bakatin"
    assert item["metadata"]["target"] == "Azerbaijan Technological University"
    assert item["metadata"]["distance"] == 3
    assert len(item["metadata"]["path"]) == 0

    # If those asserts fails, it is most likely an actual error

    # Only valid answer is 4 - Moscow
    assert dataset.score_answer(answer="4", entry=item) == 1.0
    # Selecting 8 - Russians makes you go further away from the target
    assert dataset.score_answer(answer="2", entry=item) == 0.1
    # Selecting 0 - Commmunist Party of the Soviet Union doesn't get you further away, but it doesn't get you closer either
    assert dataset.score_answer(answer="2", entry=item) == 0.1

    # Use this to check the results if you need to update this test
    # (with pytest -s)
    # for (i,_) in item['metadata']['links']:
    #    print(i, dataset.score_answer(answer=str(i), entry=item))

import pytest

from reasoning_gym.cognition.figlet_fonts import FigletFontConfig, FigletFontDataset


def test_figlet():
    """Test basic properties and solution of generated items"""
    config = FigletFontConfig(
    )
    dataset = FigletFontDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata contains required fields
        assert "font" in item["metadata"]

        # Test the scoring
        assert dataset.score_answer(answer=item['answer'], entry=item) == 1.0

def test_static_figlet():
    """Test basic properties and solution of generated items"""
    config = FigletFontConfig(
        static_word="TESTY",
        static_font="caligraphy"
    )
    dataset = FigletFontDataset(config)

    # Test partial scoring
    for item in dataset:
        assert dataset.score_answer(answer="TESTY", entry=item) == 1.0
        assert dataset.score_answer(answer="WESTY", entry=item) == 0.4
        assert dataset.score_answer(answer=None, entry=item) == 0
        break

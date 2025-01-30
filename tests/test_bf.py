import pytest

from reasoning_gym.code.bf import BFConfig, BFDataset


# def test_figlet_deterministic():
#     """Test that dataset generates same items with same seed"""
#     config = FigletFontConfig(seed=42, size=15)
#     dataset1 = FigletFontDataset(config)
#     dataset2 = FigletFontDataset(config)

#     for i in range(15):  # Only check first 15 entries for speed
#         assert dataset1[i] == dataset2[i]


def test_bf():
    """Test basic properties and solution of generated items"""
    config = BFConfig(seed=42, size=40)
    dataset = BFDataset(config)

    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item

        # Check metadata contains required fields
        assert "bfit_code" in item["metadata"]
        assert "bf_program" in item["metadata"]

        print(item["answer"])

        # Test the scoring
        assert dataset.score_answer(answer=item["answer"], entry=item) == 1.0
        assert dataset.score_answer(answer=None, entry=item) == 0.0
        assert dataset.score_answer(answer="Love is a battlefield", entry=item) == 0.01


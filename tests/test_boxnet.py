import pytest

from reasoning_gym.games import BoxnetConfig, BoxnetCurriculum, BoxnetDataset
from reasoning_gym.games.boxnet import action_from_response


def test_boxnet_config_validation():
    """Test that invalid configs raise appropriate errors"""
    with pytest.raises(AssertionError):
        config = BoxnetConfig(min_row_num=0)
        config.validate()

    with pytest.raises(AssertionError):
        config = BoxnetConfig(min_box_num=0)
        config.validate()


def test_boxnet_deterministic():
    """Test that dataset generates same items with same seed"""
    config = BoxnetConfig(seed=42, size=10)
    dataset1 = BoxnetDataset(config)
    dataset2 = BoxnetDataset(config)

    for i in range(len(dataset1)):
        assert dataset1[i]["question"] == dataset2[i]["question"]
        assert dataset1[i]["metadata"] == dataset2[i]["metadata"]


def test_boxnet_items():
    """Test basic properties of generated items"""
    config = BoxnetConfig(
        min_row_num=1, max_row_num=2, min_column_num=1, max_column_num=2, min_box_num=1, max_box_num=2, size=10, seed=42
    )
    dataset = BoxnetDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" is None or "answer" in item
        assert "metadata" in item
        assert "difficulty" in item["metadata"]
        assert "initial_state" in item["metadata"]

        # Verify row_num and column_num are within limits
        row_num = item["metadata"]["row_num"]
        column_num = item["metadata"]["column_num"]
        assert 1 <= row_num <= 2, f"row_num {row_num} outside valid range"
        assert 1 <= column_num <= 2, f"column_num {column_num} outside valid range"

        # Verify the state format
        initial_state = item["metadata"]["initial_state"]
        for key, value in initial_state.items():
            assert "_" in key, f"Invalid grid key format: {key}"
            assert isinstance(value, list), f"Grid value must be a list"
            # Check that box and target color name formats are correct
            for item_val in value:
                if item_val.startswith("box_"):
                    assert item_val[4:] in config.colour_list, f"Invalid box color: {item_val}"
                elif item_val.startswith("target_"):
                    assert item_val[7:] in config.colour_list, f"Invalid target color: {item_val}"


def test_boxnet_grid_sizes():
    """Test that generated grid respects row and column constraints"""
    config = BoxnetConfig(
        min_row_num=2,
        max_row_num=3,
        min_column_num=3,
        max_column_num=4,
        size=20,
        seed=42,
    )
    dataset = BoxnetDataset(config)

    rows_set = set()
    columns_set = set()

    for i in range(len(dataset)):
        item = dataset[i]
        row_num = item["metadata"]["row_num"]
        column_num = item["metadata"]["column_num"]

        rows_set.add(row_num)
        columns_set.add(column_num)

        # Verify grid dimensions in the state dictionary
        initial_state = item["metadata"]["initial_state"]
        grid_coords = set()
        for key in initial_state.keys():
            x, y = map(float, key.split("_"))
            grid_coords.add((x, y))

        # Check if the grid dimensions match the expected size
        unique_x = set(x for x, _ in grid_coords)
        unique_y = set(y for _, y in grid_coords)
        assert len(unique_x) == row_num, f"Expected {row_num} rows, got {len(unique_x)}"
        assert len(unique_y) == column_num, f"Expected {column_num} columns, got {len(unique_y)}"

    # With enough samples, we should see variation in grid sizes
    assert len(rows_set) > 1, "Expected variation in row numbers"
    assert len(columns_set) > 1, "Expected variation in column numbers"


def test_boxnet_box_distribution():
    """Test that boxes and targets are distributed according to configuration"""
    config = BoxnetConfig(
        min_box_num=1,
        max_box_num=2,
        colour_list=["red", "blue"],
        size=20,
        seed=43,
    )
    dataset = BoxnetDataset(config)

    for i in range(len(dataset)):
        item = dataset[i]
        initial_state = item["metadata"]["initial_state"]

        # Count boxes and targets by color
        boxes_count = {color: 0 for color in config.colour_list}
        targets_count = {color: 0 for color in config.colour_list}

        for cell_items in initial_state.values():
            for item_val in cell_items:
                if item_val.startswith("box_"):
                    color = item_val[4:]
                    boxes_count[color] += 1
                elif item_val.startswith("target_"):
                    color = item_val[7:]
                    targets_count[color] += 1

        # Verify each color has between min_box_num and max_box_num boxes
        for color in config.colour_list:
            assert (
                config.min_box_num <= boxes_count[color] <= config.max_box_num
            ), f"Color {color} has {boxes_count[color]} boxes, expected between {config.min_box_num} and {config.max_box_num}"

            # Verify the number of targets matches the number of boxes for each color
            assert (
                boxes_count[color] == targets_count[color]
            ), f"Color {color} has {boxes_count[color]} boxes but {targets_count[color]} targets"


def test_boxnet_iteration():
    """Test that iteration respects dataset size"""
    config = BoxnetConfig(size=5, seed=42)  # Small size for testing
    dataset = BoxnetDataset(config)

    # Test manual iteration
    items = []
    for item in dataset:
        items.append(item)
    assert len(items) == config.size, "Iterator should yield exactly size items"

    # Test list conversion
    items = list(dataset)
    assert len(items) == config.size, "Iterator should yield exactly size items"

    # Test multiple iterations
    first_items = list(dataset)
    second_items = list(dataset)
    assert len(first_items) == len(second_items), "Multiple iterations should yield same number of items"
    for i in range(len(first_items)):
        assert first_items[i]["question"] == second_items[i]["question"]
        assert first_items[i]["metadata"] == second_items[i]["metadata"]


def test_boxnet_curriculum():
    """Test BoxnetCurriculum functionality"""
    curriculum = BoxnetCurriculum()

    base_value = {"size": 150, "seed": 1}

    # Test initial configuration
    base_cfg = curriculum.generate_configuration(base_value)
    assert base_cfg.seed == 1
    assert base_cfg.size == 150
    assert base_cfg.min_row_num == 1 and base_cfg.max_row_num == 2
    assert base_cfg.min_column_num == 1 and base_cfg.max_column_num == 2
    assert base_cfg.min_box_num == 1 and base_cfg.max_box_num == 2

    # Test incrementing attribute levels
    curriculum.increment_attr_level("row_num")
    curriculum.increment_attr_level("box_num")

    increased_cfg = curriculum.generate_configuration(base_value)
    assert increased_cfg.min_row_num == 1 and increased_cfg.max_row_num == 3
    assert increased_cfg.min_box_num == 1 and increased_cfg.max_box_num == 3
    # Column number should remain unchanged
    assert increased_cfg.min_column_num == 1 and increased_cfg.max_column_num == 2

    # Test decrementing attribute level
    curriculum.decrement_attr_level("box_num")

    partially_decreased_cfg = curriculum.generate_configuration(base_value)
    assert partially_decreased_cfg.min_box_num == 1 and partially_decreased_cfg.max_box_num == 2
    # Row number should remain at the increased level
    assert partially_decreased_cfg.min_row_num == 1 and partially_decreased_cfg.max_row_num == 3

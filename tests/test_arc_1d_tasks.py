import random

import pytest

from reasoning_gym.arc.arc_1d_tasks import (
    task_block_and_noise_remove,
    task_block_and_noise_remove_inside,
    task_block_scale_to_dot,
    task_block_touch_dot,
    task_block_touch_dot_n_pix,
    task_change_to_five,
    task_color_left_half_blocks,
    task_copy_block_to_dots,
    task_copy_block_to_dots_colors,
    task_duplicate_block_from_seeds,
    task_fill_from_pixel,
    task_fill_until_collision,
    task_gravity,
    task_gravity_antigravity,
    task_gravity_counting,
    task_gravity_one_step,
    task_gravity_weighted_colors,
    task_identity,
    task_inverse,
    task_mark_size_two_blocks,
    task_mirror,
    task_move_block_by_own_size,
    task_move_n_pix,
    task_move_n_pix_wrapped,
    task_paint_biggest_block,
    task_recolor_blocks_by_size,
    task_recolor_blocks_from_palette,
    task_reflect_block_around_dot,
    task_reflect_block_with_border_pixel,
    task_reflect_block_with_border_pixel_random,
    task_repeat_pattern_full,
    task_sort_blocks_by_size,
    task_sort_complete_sequence,
    task_two_points_and_fill,
)


def test_all_arc_1d_tasks():
    """Test that all ARC 1D task functions can be executed without exceptions."""
    rng = random.Random(42)  # Fixed seed for reproducibility
    size = 20  # Reasonable size for testing

    # Test all task functions
    # Fixed move_pix value for testing
    move_pix = 2

    # Test task augmentation functions
    base_task = task_move_n_pix(rng, size, move_pix, True)
    assert base_task is not None

    mirrored = task_mirror(base_task)
    assert mirrored is not None
    assert mirrored["input"] == list(reversed(base_task["input"]))
    assert mirrored["output"] == list(reversed(base_task["output"]))

    inversed = task_inverse(base_task)
    assert inversed is not None
    assert inversed["input"] == base_task["output"]
    assert inversed["output"] == base_task["input"]

    identical = task_identity(base_task)
    assert identical is not None
    assert identical == base_task

    tasks = [
        (task_move_n_pix, {"move_pix": move_pix, "solid": True}),
        (task_move_n_pix_wrapped, {"move_pix": move_pix, "solid": True}),
        (task_gravity, {}),
        (task_gravity_counting, {}),
        (task_gravity_antigravity, {}),
        (task_block_touch_dot, {}),
        (task_block_touch_dot_n_pix, {"move_pix": move_pix}),
        (task_block_scale_to_dot, {}),
        (task_two_points_and_fill, {}),
        (task_reflect_block_with_border_pixel, {}),
        (task_reflect_block_with_border_pixel_random, {}),
        (task_reflect_block_around_dot, {}),
        (task_block_and_noise_remove, {}),
        (task_block_and_noise_remove_inside, {}),
        (task_copy_block_to_dots, {}),
        (task_copy_block_to_dots_colors, {}),
        (task_paint_biggest_block, {}),
        (task_sort_blocks_by_size, {}),
        (task_sort_complete_sequence, {}),
        (task_recolor_blocks_by_size, {}),
        (task_gravity_one_step, {}),
        (task_move_block_by_own_size, {}),
        (task_change_to_five, {}),
        (task_recolor_blocks_from_palette, {}),
        (task_duplicate_block_from_seeds, {}),
        (task_fill_from_pixel, {}),
        (task_mark_size_two_blocks, {}),
        (task_fill_until_collision, {}),
        (task_repeat_pattern_full, {}),
        (task_gravity_weighted_colors, {}),
        (task_color_left_half_blocks, {}),
    ]

    for task_func, kwargs in tasks:
        # Try multiple times as some functions might return None for certain inputs
        success = False
        for _ in range(10):  # Try up to 10 times
            try:
                result = task_func(rng, size, **kwargs)
                if result is not None:
                    success = True
                    # Basic structure checks
                    assert isinstance(result, dict)
                    assert "input" in result
                    assert "output" in result
                    assert len(result["input"]) == size
                    assert len(result["output"]) == size
                    break
            except Exception as e:
                pytest.fail(f"Task {task_func.__name__} failed with error: {str(e)}")

        assert success, f"Task {task_func.__name__} always returned None in 10 attempts"

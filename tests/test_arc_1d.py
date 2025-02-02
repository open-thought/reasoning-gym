import random
import pytest
from reasoning_gym.cognition.arc_1d import (
    task_move_n_pix, task_move_n_pix_wrapped, task_gravity, task_gravity_counting,
    task_gravity_antigravity, task_block_touch_dot, task_block_touch_dot_n_pix,
    task_block_scale_to_dot, task_two_points_and_fill, task_reflect_block_with_border_pixel,
    task_reflect_block_with_border_pixel_random, task_reflect_block_around_dot,
    task_block_and_noise_remove, task_block_and_noise_remove_inside,
    task_copy_block_to_dots, task_copy_block_to_dots_colors, task_paint_biggest_block,
    task_sort_blocks_by_size, task_sort_complete_sequence, task_recolor_blocks_by_size,
    task_gravity_one_step, task_move_block_by_own_size, task_change_to_five,
    task_recolor_blocks_from_palette, task_duplicate_block_from_seeds,
    task_fill_from_pixel, task_mark_size_two_blocks, task_fill_until_collision,
    task_repeat_pattern_full, task_gravity_weighted_colors, task_color_left_half_blocks
)

def test_all_arc_1d_tasks():
    """Test that all ARC 1D task functions can be executed without exceptions."""
    rng = random.Random(42)  # Fixed seed for reproducibility
    size = 20  # Reasonable size for testing
    
    # Test all task functions
    tasks = [
        (task_move_n_pix, {"move_pix": 2, "solid": True}),
        (task_move_n_pix_wrapped, {"move_pix": 2, "solid": True}),
        (task_gravity, {}),
        (task_gravity_counting, {}),
        (task_gravity_antigravity, {}),
        (task_block_touch_dot, {}),
        (task_block_touch_dot_n_pix, {"move_pix": 2}),
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
        (task_color_left_half_blocks, {})
    ]

    for task_func, kwargs in tasks:
        # Try multiple times as some functions might return None for certain inputs
        success = False
        for _ in range(10):  # Try up to 10 times
            try:
                result = task_func(size, rng, **kwargs)
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

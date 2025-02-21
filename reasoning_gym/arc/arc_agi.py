from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable, Optional

import arckit

from reasoning_gym.arc.board_format import (
    ARC_PROMPT_TEMPLATE,
    BoardFormattingOptions,
    format_board,
    format_board_pair,
    parse_board,
)
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.factory import register_dataset


@dataclass
class ArcAgiConfig:
    use_train: bool = True
    use_eval: bool = True
    board_format_opts: BoardFormattingOptions = field(default_factory=lambda: BoardFormattingOptions())

    # Augmentation options
    rotations: list[str] = field(default_factory=lambda: ["90", "180", "270"])  # empty list for no rotations
    mirrors: list[str] = field(
        default_factory=lambda: ["horizontal", "vertical", "diagonal", "counterdiagonal"]
    )  # empty list for no mirrors
    use_color_permutation: bool = True
    shuffle_example_order: bool = True  # whether to shuffle the order of example board pairs for each riddle

    seed: Optional[int] = None
    size: int = 500

    def validate(self):
        assert self.size > 0, "Size of dataset must be positive."
        valid_rotations = ["90", "180", "270"]
        valid_mirrors = ["horizontal", "vertical", "diagonal", "counterdiagonal"]
        for rot in self.rotations:
            assert rot in valid_rotations, f"Invalid rotation option: {rot}"
        for mirror in self.mirrors:
            assert mirror in valid_mirrors, f"Invalid mirror option: {mirror}"


Board = list[list[int]]


def identity(board: Board) -> Board:
    return board


def rot90(board: Board) -> Board:
    """quarter clockwise rotation"""
    return [row for row in zip(*board[::-1])]


def rot180(board: Board) -> Board:
    """half rotation"""
    return [row[::-1] for row in board[::-1]]


def rot270(board: Board) -> Board:
    """quarter anticlockwise rotation"""
    return [row[::-1] for row in zip(*board[::-1])][::-1]


def hmirror(board: Board) -> Board:
    """mirroring along horizontal"""
    return board[::-1]


def vmirror(board: Board) -> Board:
    """mirroring along vertical"""
    return [row[::-1] for row in board]


def dmirror(board: Board) -> Board:
    """mirroring along diagonal"""
    return list(zip(*board))


def cmirror(board: Board) -> Board:
    """mirroring along counterdiagonal"""
    return list(zip(*[r[::-1] for r in board[::-1]]))


def cmap(board: Board, colors: list[int]) -> Board:
    return [[colors[c] for c in row] for row in board]


# ROTATION_AUGMENTATIONS = [identity, rot90, rot180, rot270]
# MIRROR_AUGMENTATIONS = [identity, hmirror, vmirror, dmirror, cmirror]


class ArcAgiDataset(ProceduralDataset):
    def __init__(self, config: ArcAgiConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)
        self.board_format_opts = config.board_format_opts
        self._prompt_templates = ARC_PROMPT_TEMPLATE

        self._tasks = {}
        train_set, eval_set = arckit.load_data()
        if config.use_train:
            for x in train_set:
                self._tasks[x.id] = x.to_dict()
        if config.use_eval:
            for x in eval_set:
                self._tasks[x.id] = x.to_dict()
        self._task_ids = list(self._tasks.keys())

    def _create_augmentation_fn(self, rng: Random) -> Callable[[Board], Board]:
        """Create a composite augmentation function from enabled options"""
        fns = []

        # Map rotation strings to functions
        rotation_map = {"90": rot90, "180": rot180, "270": rot270}
        if self.config.rotations:
            chosen_rot = rng.choice([identity] + [rotation_map[r] for r in self.config.rotations])
            fns.append(chosen_rot)

        # Map mirror strings to functions
        mirror_map = {"horizontal": hmirror, "vertical": vmirror, "diagonal": dmirror, "counterdiagonal": cmirror}
        if self.config.mirrors:
            chosen_mirror = rng.choice([identity] + [mirror_map[m] for m in self.config.mirrors])
            fns.append(chosen_mirror)

        if self.config.use_color_permutation:
            color_table = list(range(10))
            rng.shuffle(color_table)
            fns.append(lambda x: cmap(x, color_table))

        def composite_fn(board: Board) -> Board:
            result = board
            for fn in fns:
                result = fn(result)
            return result

        return composite_fn

    def __getitem__(self, idx: int) -> dict:
        """
        Generate a single ARC-AGI-1 task
        """
        rng = Random(self.seed + idx)

        task_id = rng.choice(self._task_ids)
        task = self._tasks[task_id]

        # Create augmentation function to be used for all examples
        augment = self._create_augmentation_fn(rng)

        train = task["train"]
        test = task["test"][0]

        # Apply augmentation to all train examples
        augmented_train = []
        for p in train:
            augmented_train.append({"input": augment(p["input"]), "output": augment(p["output"])})

        if self.config.shuffle_example_order:
            rng.shuffle(augmented_train)

        examples = [
            format_board_pair(i + 1, p, formatting_options=self.config.board_format_opts)
            for i, p in enumerate(augmented_train)
        ]
        examples = "".join(examples)

        # Apply augmentation to test example
        augmented_test_input = augment(test["input"])
        augmented_test_output = augment(test["output"])

        test_input = format_board(augmented_test_input, self.board_format_opts)
        test_output = format_board(augmented_test_output, self.board_format_opts)

        input_prompt = self._prompt_templates.format(examples=examples, input_grid=test_input)

        def totuple(board: list[list[int]]) -> tuple[tuple[int, ...], ...]:
            return tuple(tuple(r) for r in board)

        return {
            "question": input_prompt,
            "answer": test_output,
            "metadata": {
                "input": totuple(augmented_test_input),
                "output": totuple(augmented_test_output),
                "task_id": task_id,
            },
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        reward = 0.0
        metadata = entry["metadata"]
        if answer is not None:
            try:
                answer_board = parse_board(answer, self.board_format_opts)
                if answer_board == metadata["output"]:
                    reward = 1.0
                else:
                    reward = 0.05
            except:
                reward = 0.01
        return reward


register_dataset("arc_agi", ArcAgiDataset, ArcAgiConfig)

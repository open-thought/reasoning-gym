from dataclasses import dataclass, field
from random import Random
from typing import Any, Optional

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
    seed: Optional[int] = None
    size: int = 500

    def validate(self):
        assert self.size > 0, "Size of dataset must be positive."


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

    def __getitem__(self, idx: int) -> dict:
        """
        Generate a single ARC-AGI-1 task
        """
        rng = Random(self.seed + idx)

        task_id = rng.choice(self._task_ids)
        task = self._tasks[task_id]

        train = task["train"]
        test = task["test"][0]

        examples = [
            format_board_pair(i + 1, p, formatting_options=self.config.board_format_opts) for i, p in enumerate(train)
        ]
        examples = "".join(examples)
        test_input = format_board(test["input"], self.board_format_opts)
        test_output = format_board(test["output"], self.board_format_opts)

        input_prompt = self._prompt_templates.format(examples=examples, input_grid=test_input)

        def totuple(board: list[list[int]]) -> tuple[tuple[int, ...], ...]:
            return tuple(tuple(r) for r in board)

        return {
            "question": input_prompt,
            "answer": test_output,
            "metadata": {
                "input": totuple(test["input"]),
                "output": totuple(test["output"]),
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


if __name__ == "__main__":
    cfg = ArcAgiConfig(seed=99)
    test = ArcAgiDataset(cfg)

    x = test[1]

    a = """1 6 7
6 7 6
2 2 6"""

    print("q:", x["question"])
    print("a:", x["answer"])
    print("score:", test.score_answer(answer=a, entry=x))

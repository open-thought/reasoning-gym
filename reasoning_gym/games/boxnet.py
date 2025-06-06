import copy
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..coaching import BaseCurriculum, RangeAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

BOXNET_PROMPT = """
You are a central planner tasked with directing agents in a grid-like field to move colored boxes to their corresponding color-coded targets.
Each agent occupies a 1x1 square and can only interact with objects within its square. Agents can move a box to an adjacent square or
directly to a target square of the same color. A square may contain multiple boxes and targets. The squares are identified by their center
coordinates (e.g., square[0.5, 0.5]). Actions are formatted as: move(box_color, destination), where box_color is the color of the box and
destination is either a target of the same color or an adjacent square. Your objective is to create a sequence of action plans that instructs
each agent to match all boxes to their color-coded targets in the most efficient manner.

Please adhere to the following rules when specifying your action plan:
1. Single Action per Agent: Assign only one action to each agent at a time. However, the final answer shoule be a list of action plans for multiple steps.
2. Unique Agent Keys: Use unique keys for each agent in the JSON format action plan. The key should be the agent's coordinates in the format "Agent[x, y]".
3. Prioritize Matching Boxes to Targets: Always prioritize actions that will match a box to its target over moving a box to an adjacent square.
4. Sequential Action Planning: The whole returned answer should be a list of action plans for multiple steps, do not just return one step plan.
5. Clear Formatting: Ensure the action plan is clearly formatted in JSON, with each agent's action specified as a key-value pair.
6. Conflict Resolution: Ensure that no two agents are assigned actions that would interfere with each other.
7. Optimize Efficiency: Aim to minimize the number of moves required to match all boxes with their targets.

Here is the format for your action plan:
Please provide your final answer as a list of action dictionaries.
For example:
```json
[{"Agent[0.5, 0.5]":"move(box_blue, square[0.5, 1.5])", "Agent[1.5, 0.5]":"move(box_red, target_red)"}, {"Agent[0.5, 1.5]":"move(box_blue, target_blue)", "Agent[2.5, 0.5]":"move...}, {...}...]
```
Include an agent in the action plan only if it has a task to perform next.
"""

DATASET_NAME = "boxnet"


def action_from_response(pg_dict_input, original_response_dict_list):
    pg_dict_current = copy.deepcopy(pg_dict_input)

    for original_response_dict in original_response_dict_list:
        transformed_dict = {}
        for key, value in original_response_dict.items():
            coordinates = tuple(map(float, re.findall(r"\d+\.?\d*", key)))
            match = re.match(r"move\((.*?),\s(.*?)\)", value)
            if match:
                item, location = match.groups()
                if "square" in location:
                    location = tuple(map(float, re.findall(r"\d+\.?\d*", location)))
                transformed_dict[coordinates] = [item, location]

        # Process each move with the current state
        for key, value in transformed_dict.items():
            current_pos = f"{key[0]}_{key[1]}"

            # Check if current pos is not in the grid, i.e. the LLM hallucinated a non-existent position
            if current_pos not in pg_dict_current:
                # For now, we just skip these invalid moves, but it may be desirable to also penalise them somehow
                continue

            # Check if this is a box-target matching move
            if (
                value[0] in pg_dict_current[current_pos]
                and isinstance(value[1], str)
                and value[1] in pg_dict_current[current_pos]
                and value[0].startswith("box_")
                and value[1].startswith("target_")
                and value[0][4:] == value[1][7:]
            ):
                # Remove both box and target when matched
                pg_dict_current[current_pos].remove(value[0])
                pg_dict_current[current_pos].remove(value[1])

            # Check if this is a movement to another square
            elif value[0] in pg_dict_current[current_pos] and isinstance(
                value[1], tuple
            ):  # Only check coordinates for square movements
                # Calculate if move is to adjacent square
                if (np.abs(key[0] - value[1][0]) == 0 and np.abs(key[1] - value[1][1]) == 1) or (
                    np.abs(key[0] - value[1][0]) == 1 and np.abs(key[1] - value[1][1]) == 0
                ):
                    # Move box to new location
                    target_pos = f"{value[1][0]}_{value[1][1]}"
                    pg_dict_current[current_pos].remove(value[0])
                    pg_dict_current[target_pos].append(value[0])
    return pg_dict_current


@dataclass
class BoxnetConfig:
    """Configuration for Boxnet task generation"""

    min_row_num: int = 1
    max_row_num: int = 4
    min_column_num: int = 2
    max_column_num: int = 4
    min_box_num: int = 1
    max_box_num: int = 1
    colour_list: list[str] = field(default_factory=lambda: ["red", "blue", "green"])
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.size > 0, "size must be greater than 0"
        assert self.min_row_num > 0, "min_row_num must be greater than 0"
        assert self.max_row_num > 0, "max_row_num must be greater than 0"
        assert self.min_column_num > 0, "min_column_num must be greater than 0"
        assert self.max_column_num > 0, "max_column_num must be greater than 0"
        assert self.min_box_num > 0, "min_box_num must be greater than 0"
        assert self.max_box_num > 0, "max_box_num must be greater than 0"
        assert self.min_box_num <= self.max_box_num, "min_box_num must be less than or equal to max_box_num"


class BoxnetDataset(ProceduralDataset):
    def __init__(self, config: BoxnetConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + idx)
        row_num = rng.randint(self.config.min_row_num, self.config.max_row_num)
        column_num = rng.randint(self.config.min_column_num, self.config.max_column_num)
        pg_dict = self._generate_boxnet(rng, row_num, column_num, self.config.colour_list)
        pg_dict_initial = copy.deepcopy(pg_dict)

        state_update_prompt = self.state_update_func(row_num, column_num, pg_dict_initial)
        question = BOXNET_PROMPT + "\n\n" + "The current left boxes and agents are: " + state_update_prompt + "\n"
        return {
            "question": question,
            "answer": None,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "row_num": row_num,
                "column_num": column_num,
                "initial_state": pg_dict,
                "difficulty": {
                    "row_num": (self.config.min_row_num, self.config.max_row_num),
                    "column_num": (self.config.min_column_num, self.config.max_column_num),
                    "box_num": (self.config.min_box_num, self.config.max_box_num),
                },
            },
        }

    def _generate_boxnet(self, rng: random.Random, row_num: int, column_num: int, colour_list: list[str]):
        """Generate a Boxnet task"""
        pg_dict = {}
        for i in range(row_num):
            for j in range(column_num):
                pg_dict[str(i + 0.5) + "_" + str(j + 0.5)] = []

        for colour in colour_list:
            box_num = rng.randint(self.config.min_box_num, self.config.max_box_num)
            for _ in range(box_num):
                N_box = rng.randint(0, row_num * column_num - 1)
                a_box = N_box // column_num
                b_box = N_box % column_num
                N_target = rng.randint(0, row_num * column_num - 1)
                a_target = N_target // column_num
                b_target = N_target % column_num
                pg_dict[str(a_box + 0.5) + "_" + str(b_box + 0.5)].append("box_" + colour)
                pg_dict[str(a_target + 0.5) + "_" + str(b_target + 0.5)].append("target_" + colour)
        return pg_dict

    def surround_index_func(self, row_num, coloum_num, row_index, coloum_index):
        surround_index_list = []
        for i, j in (
            [row_index - 1, coloum_index],
            [row_index + 1, coloum_index],
            [row_index, coloum_index - 1],
            [row_index, coloum_index + 1],
        ):
            if (
                i >= 0
                and i <= row_num - 1
                and j >= 0
                and j <= coloum_num - 1
                and not (i == row_index and j == coloum_index)
            ):
                surround_index_list.append([i + 0.5, j + 0.5])
        return surround_index_list

    def state_update_func(self, pg_row_num, pg_column_num, pg_dict):
        state_update_prompt = ""
        for i in range(pg_row_num):
            for j in range(pg_column_num):
                square_item_list = pg_dict[str(i + 0.5) + "_" + str(j + 0.5)]
                square_item_only_box = [item for item in square_item_list if item[:3] == "box"]
                surround_index_list = self.surround_index_func(pg_row_num, pg_column_num, i, j)
                state_update_prompt += f"Agent[{i+0.5}, {j+0.5}]: I am in square[{i+0.5}, {j+0.5}], I can observe {square_item_list}, I can do "
                action_list = []
                for box in square_item_only_box:
                    for surround_index in surround_index_list:
                        action_list.append(f"move({box}, square{surround_index})")
                    if "target" + box[3:] in square_item_list:
                        action_list.append(f"move({box}, target{box[3:]})")
                state_update_prompt += f"{action_list}\n"
        return state_update_prompt

    def score_answer(self, answer: str | None, entry: dict[str, Any]) -> float:
        reward = 0.0
        if answer is not None:
            try:
                answer_dict = json.loads(answer)
            except:
                return 0.00

            pg_dict_returned = action_from_response(entry["metadata"]["initial_state"], answer_dict)

            initial_boxes = 0
            for items in entry["metadata"]["initial_state"].values():
                initial_boxes += sum(1 for item in items if item.startswith("box_"))

            remaining_boxes = 0
            for items in pg_dict_returned.values():
                remaining_boxes += sum(1 for item in items if item.startswith("box_"))

            lifted_ratio = (initial_boxes - remaining_boxes) / initial_boxes
            reward = max(0.05, lifted_ratio)

        return reward


class BoxnetCurriculum(BaseCurriculum):
    """Curriculum for Boxnet"""

    def __init__(self):
        super().__init__(BoxnetCurriculum.__name__, BoxnetConfig)
        self._define_attributes(
            RangeAttributeDefinition(
                name="row_num",
                description="The maximum number of rows in the grid",
                lower_field_name="min_row_num",
                upper_field_name="max_row_num",
                levels=list(range(1, 10)),
                ensure_interval=True,
            ),
            RangeAttributeDefinition(
                name="column_num",
                description="The maximum number of columns in the grid",
                lower_field_name="min_column_num",
                upper_field_name="max_column_num",
                levels=list(range(1, 10)),
                ensure_interval=True,
            ),
            RangeAttributeDefinition(
                name="box_num",
                description="The maximum number of boxes in the grid",
                lower_field_name="min_box_num",
                upper_field_name="max_box_num",
                levels=list(range(1, 10)),
                ensure_interval=True,
            ),
        )


register_dataset(DATASET_NAME, BoxnetDataset, BoxnetConfig, BoxnetCurriculum)

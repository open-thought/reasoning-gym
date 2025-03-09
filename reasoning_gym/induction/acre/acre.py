"""ACRE(Abstract Causal REasoning Beyond Covariation) dataset"""

# Culled and Adapted from https://github.com/WellyZhang/ACRE
# Licensed under GPL-3.0

from dataclasses import dataclass
from random import Random
from typing import Optional

from reasoning_gym.factory import ProceduralDataset, register_dataset

from .blicket import config_control, dist_control, final_parse, serialize
from .const import ALL_CONFIG_SIZE, ATTR_CONFIG_SIZE


# Create blicket questions
@dataclass
class ACREDatasetConfig:
    """Configuration for ACRE dataset generation"""

    train: int = 1  # The default is 1 for training, otherwise 0 for validation and testing
    size: int = 500  #  Split ratio = 6 : 2 : 2 -> IID : Comp : Sys
    seed: Optional[int] = None

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert self.train in (0, 1), "train must be either 0 or 1"
        assert self.size > 0, "Dataset size must be positive."


class ACREDataset(ProceduralDataset):

    def __init__(self, config: ACREDatasetConfig):
        super().__init__(config, config.seed, config.size)
        self.questions = self._generate_questions()
        self.prompt_template = """You are a researcher studying causal relationships using Blicket experiments. In these experiments, certain objects (called 'blickets') have the hidden property of activating a detector, causing its light to turn on.

Each example shows the results of placing different combinations of objects on the detector. Each object is described by color, material and shape. Your task is to determine whether a new combination of objects will cause the detector to activate.

After observing the previous examples, respond with:
- "on" if you can determine the detector light will turn on
- "off" if you can determine the detector light will stay off
- "undetermined" if there is insufficient evidence to reach a conclusion

Do not use quotation marks in your answer.

Previous experimental results:
{examples}

New test case:
{input}

What is the detector light status?"""

    def _generate_questions(self):
        """
        Generates questions of particular size
        """

        iid_size = int(0.6 * self.config.size)
        comp_size = int(0.2 * self.config.size)
        sys_size = self.config.size - (iid_size + comp_size)
        rng = Random(self.seed)
        iid_questions = config_control(iid_size, self.config.train, ALL_CONFIG_SIZE, "IID", rng)
        comp_questions = config_control(comp_size, self.config.train, ATTR_CONFIG_SIZE, "Comp", rng)
        sys_questions = dist_control(sys_size, self.config.train, "Sys", rng)

        questions = []
        questions.extend(iid_questions)
        questions.extend(comp_questions)
        questions.extend(sys_questions)
        rng.shuffle(questions)
        final_questions = final_parse(serialized_questions=serialize(questions))
        return final_questions

    def __getitem__(self, idx: int) -> dict:
        """Generate a single induction-based list function dataset"""
        input = self.questions[idx]
        examples = input["examples"]
        formatted_examples = ""
        for object in examples:
            input_ = ", ".join(" ".join(x) for x in object["input"])
            output = object["output"]
            if len(formatted_examples) > 0:
                formatted_examples += "\n"
            formatted_examples += f"{input_} → {output}"

        prompt_input = ", ".join(" ".join(x) for x in input["question"]["input"])
        answer = input["question"]["output"]
        question = self.prompt_template.format(examples=formatted_examples, input=prompt_input)
        return {"question": question, "answer": answer, "metadata": {}}


register_dataset("acre", ACREDataset, ACREDatasetConfig)

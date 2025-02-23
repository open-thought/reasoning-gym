# TODO: consider whether this belongs in the "code" directory
import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Optional

from ..factory import ProceduralDataset, register_dataset

OUTPUT_PREDICTION_PROMPT_TEMPLATE = """
You are given a question that requires some input and output variables as follows:

{0}

The input and output requirements are as follows:

{1}

Given the following input:

{2}

Can you predict the output without writing any code? Please think and then provide only the exact output as your final answer, which should strictly match the output requirement as specified.

Tip: Here is a reference code snippet for this question. You can refer to this code to guide your reasoning but not copy spans of code directly.

{3}
"""

INPUT_PREDICTION_PROMPT_TEMPLATE = """
You are given a question that requires some input and output variables as follows:

{0}

The input and output requirements are as follows:

{1}

Given the following output:

{2}

Can you predict a feasible input without writing any code? Please reason and put your final answer in the following json format: "input": <your input>, where <your input> should be a dictionary, even if the there is only one input variable, with keys strictly matching the input variables' names as specified.

Tip: Here is a reference code snippet for this question. You can refer to this code to guide your reasoning but not copy spans of code directly.

{3}
"""

# TODO: also add input prediction prompt


@dataclass
class CodeIOConfig:
    """Configuration for BF task generation"""

    seed: Optional[int] = None
    size: int = 500
    input_prediction_probability: float = 0.5

    def validate(self) -> None:
        """Validate configuration parameters"""
        pass


class CodeIODataset(ProceduralDataset):
    def __init__(self, config: CodeIOConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __len__(self) -> int:
        return self.config.size

    def __iter__(self):
        self._current_idx = 0
        return self

    def __next__(self):
        if self._current_idx >= self.config.size:
            raise StopIteration
        item = self[self._current_idx]
        self._current_idx += 1
        return item

    def _generate_io_pairs(self, main_code: str, input_generator_code: str, num_pairs: int = 1):
        local_vars = {}
        exec(main_code, {}, local_vars)
        exec(input_generator_code, {}, local_vars)
        io_pairs = []
        for _ in range(num_pairs):
            inputs = local_vars["input_generator"]()
            outputs = local_vars["main"](**inputs)
            io_pairs.append((inputs, outputs))
        return io_pairs

    def __getitem__(self, idx: int) -> dict:
        """Generate a single CodeI/O reasoning task"""
        rng = Random(self.seed + idx)

        # TODO: load data from external source (HuggingFace dataset?)
        jsonl_path = Path("data/codeio.jsonl")

        # Avoid loading the entire file into memory in case it's large
        with open(jsonl_path, "r", encoding="utf-8") as f:
            num_lines = sum(1 for _ in f)
            random_line_number = rng.randint(0, num_lines - 1)

            f.seek(0)
            for current_line_number, line in enumerate(f):
                if current_line_number == random_line_number:
                    json_data = json.loads(line.strip())

        query = json_data["query"]
        parameters = json_data["parameters"]
        reference_code = json_data["reference_code"]
        input_generator_code = json_data["input_generator"]

        input_data, output_data = self._generate_io_pairs(reference_code, input_generator_code, num_pairs=1)[0]

        if rng.random() < self.config.input_prediction_probability:
            question = OUTPUT_PREDICTION_PROMPT_TEMPLATE.format(query, parameters, input_data, reference_code)
            solution = output_data
        else:
            question = INPUT_PREDICTION_PROMPT_TEMPLATE.format(query, parameters, output_data, reference_code)
            solution = input_data

        return {
            "question": question,
            "answer": solution,
            "metadata": {},
        }

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        # TODO: better answer scoring
        oracle_answer = entry["answer"].strip()
        reward = 0.0
        if answer is not None and len(answer) > 0:
            answer = answer.strip()
            if answer == oracle_answer:
                reward = 1.0
            elif oracle_answer in answer:
                reward = len(oracle_answer) / len(answer)
            else:
                reward = 0.01

        return reward


# Register the dataset
register_dataset("codeio", CodeIODataset, CodeIOConfig)

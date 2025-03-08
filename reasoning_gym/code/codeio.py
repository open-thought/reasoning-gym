import gzip
import json
from dataclasses import dataclass
from random import Random
from typing import Any, Optional

import zss

from ..data import get_data_file_path
from ..factory import ProceduralDataset, register_dataset

OUTPUT_PREDICTION_PROMPT_TEMPLATE = """
You are given a question that requires some input and output variables as follows:

{0}

The input and output requirements are as follows:

{1}

Given the following input:

{2}

Can you predict the output without writing any code? Please think and then provide the exact output in the form of a JSON object as your final answer. The keys and values of the object should strictly match the output requirement as specified.

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

Can you predict a feasible input without writing any code? Please reason and put your final answer in the form of a JSON object, even if the there is only one input variable, with keys strictly matching the input variables' names as specified.

Tip: Here is a reference code snippet for this question. You can refer to this code to guide your reasoning but not copy spans of code directly.

{3}
"""


@dataclass
class CodeIOConfig:
    """Configuration for CodeI/O reasoning task generation"""

    seed: Optional[int] = None
    size: int = 500
    input_prediction_probability: float = 0.5

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert 0.0 <= self.input_prediction_probability <= 1.0, "input_prediction_probability must be in [0, 1]"


class CodeIODataset(ProceduralDataset):
    """
    Exercise some caution when using this dataset, as it involves executing arbitrary code snippets.
    These code snippets are transformed by an LLM from raw code files which have been curated from high-quality sources.
    However, there is still a risk that the LLM could have introduced code with bad effects.
    """

    _jsonl_data: Optional[list] = None

    def __init__(self, config: CodeIOConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

        self._data_path = get_data_file_path("codeio.jsonl.gz")

        with gzip.open(self._data_path, "rt", encoding="utf-8") as f:
            CodeIODataset._jsonl_data = [json.loads(line) for line in f]

    def _generate_io_pair(self, main_code: str, input_generator_code: str, rng: Random, max_retries: int = 3):
        local_vars = {}
        exec(main_code, {"Random": Random}, local_vars)
        exec(input_generator_code, {"Random": Random}, local_vars)
        for _ in range(max_retries):
            try:
                inputs = local_vars["generate_inputs"](rng)
                outputs = local_vars["main_solution"](**inputs)
            except Exception:
                # Retry
                continue
            return inputs, outputs
        return {}, {}

    def __getitem__(self, idx: int) -> dict:
        """Generate a single CodeI/O reasoning task"""
        rng = Random(self.seed + idx)

        json_data = rng.choice(CodeIODataset._jsonl_data)

        query = json_data["task_description"]
        parameters = json_data["input_output_spec"]
        reference_code = json_data["code_sample"]
        input_generator_code = json_data["input_generator"]

        input_data, output_data = self._generate_io_pair(reference_code, input_generator_code, rng)

        if rng.random() < self.config.input_prediction_probability:
            question = OUTPUT_PREDICTION_PROMPT_TEMPLATE.format(query, parameters, input_data, reference_code)
            solution = json.dumps(output_data)
        else:
            question = INPUT_PREDICTION_PROMPT_TEMPLATE.format(query, parameters, output_data, reference_code)
            solution = json.dumps(input_data)

        return {
            "question": question,
            "answer": solution,
            "metadata": {"input_data": input_data, "output_data": output_data},
        }

    def _json_to_tree(self, data, label="root"):
        """Recursively convert a JSON dictionary to a ZSS tree."""
        if isinstance(data, dict):
            node = zss.Node(label)
            for key, value in sorted(data.items()):
                node.addkid(self._json_to_tree(value, key))
            return node
        elif isinstance(data, list):
            node = zss.Node(label)
            for idx, item in enumerate(data):
                node.addkid(self._json_to_tree(item, f"item_{idx}"))
            return node
        else:
            return zss.Node(f"{label}:{data}")

    def _compute_json_similarity(self, json1, json2):
        """Compute a similarity score in [0, 1] between two JSON dictionaries using tree edit distance."""
        tree1 = self._json_to_tree(json1)
        tree2 = self._json_to_tree(json2)

        def _str_edit_distance(str1, str2):
            """Compute Levenshtein edit distance between two strings."""
            m, n = len(str1), len(str2)
            prev = list(range(n + 1))
            curr = [0] * (n + 1)
            for i in range(1, m + 1):
                curr[0] = i
                for j in range(1, n + 1):
                    if str1[i - 1] == str2[j - 1]:
                        curr[j] = prev[j - 1]
                    else:
                        curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
                prev, curr = curr, prev
            return prev[n]

        def _tree_node_edit_distance(text1: str, text2: str):
            """Compute edit distance between two tree nodes based on their types."""
            if ":" not in text1 or ":" not in text2:
                return _str_edit_distance(text1, text2)

            key1, value1 = text1.split(":", 1)
            key2, value2 = text2.split(":", 1)

            key_dist = _str_edit_distance(key1, key2) if key1 != key2 else 0
            value_dist = _str_edit_distance(value1, value2) if value1 != value2 else 0

            if value1 != value2:
                # Numeric, allowing decimals
                if value1.replace(".", "").isnumeric() and value2.replace(".", "").isnumeric():
                    try:
                        # TODO: Consider a more sophisticated distance metric for numeric values?
                        abs1, abs2 = abs(float(value1)), abs(float(value2))
                        divisor = max(min(abs1, abs2), 10e-5)
                        value_dist += (abs1 - abs2) / divisor
                    except ValueError:
                        # Fall back on string edit distance
                        pass
                elif value1.isnumeric() or value2.isnumeric():
                    # Penalise severely if the answer is numeric when it shouldn't be, or vice versa
                    value_dist += max(len(text1), len(text2))

            return key_dist + value_dist

        edit_distance = zss.simple_distance(tree1, tree2, label_dist=_tree_node_edit_distance)
        max_size = max(len(json.dumps(json1)), len(json.dumps(json2)))

        similarity_score = 1 - (edit_distance / (0.2 * max_size))
        return max(0, similarity_score)

    def _score_answer_json(self, answer_json: dict, oracle_json: dict, max_score: float) -> float:
        """If the answer is valid JSON, compute a similarity score between the answer and the oracle JSON."""
        if answer_json == oracle_json:
            return max_score
        else:
            similarity = self._compute_json_similarity(answer_json, oracle_json)
            # 0.01 minimum reward, since it produced a valid JSON output
            return max(similarity * max_score, 0.01)

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        oracle_answer = entry["answer"].strip()
        reward = 0.0
        if answer is not None and len(answer) > 0:
            answer = answer.strip()
            if answer == oracle_answer:
                reward = 1.0
            elif "{" in answer and "}" in answer:
                # Check if the answer contains a correct format JSON object somewhere
                # But penalise for length & accuracy
                ans_first_open, ans_last_close = answer.index("{"), answer.rindex("}")
                extra_chars = len(answer[:ans_first_open]) + len(answer[ans_last_close + 1 :])

                # 0.5 is arbitrary here, but the answers are very short so it seems harsh to penalize too much
                # e.g. if oracle is {"steps": "3"} and answer is "The correct answer is: {"steps": "3"}"
                max_score = max(len(oracle_answer) / (len(oracle_answer) + 0.5 * extra_chars), 0.2)

                try:
                    answer_dict = json.loads(answer[ans_first_open : ans_last_close + 1])
                    oracle_dict = json.loads(oracle_answer)
                    return self._score_answer_json(answer_dict, oracle_dict, max_score)
                except json.JSONDecodeError:
                    if oracle_answer in answer:
                        reward = len(oracle_answer) / len(answer)
                    else:
                        reward = 0.00
            else:
                reward = 0.00

        return reward


# Register the dataset
register_dataset("codeio", CodeIODataset, CodeIOConfig)

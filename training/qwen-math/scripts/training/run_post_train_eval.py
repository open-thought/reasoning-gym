# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom evaluation tasks for LightEval."""

import numpy as np
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    compare_gold_target,
    extract_target_from_pred,
    get_extraction_regexes,
)
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import PassAtK
from lighteval.metrics.utils.metric_utils import MetricCategory, MetricUseCase, SampleLevelMetric
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language

# Prompt template adapted from
# - simple-evals: https://github.com/openai/simple-evals/blob/6e84f4e2aed6b60f6a0c7b8f06bbbf4bfde72e58/math_eval.py#L17
# - Llama 3: https://huggingface.co/datasets/meta-llama/Llama-3.2-1B-Instruct-evals/viewer/Llama-3.2-1B-Instruct-evals__math__details?views%5B%5D=llama_32_1b_instruct_evals__math__details
# Note that it is important to have the final answer in a box for math-verify to work correctly
MATH_QUERY_TEMPLATE = """
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
""".strip()


math_pass_at_1_2n = SampleLevelMetric(
    metric_name="math_pass@1:2_samples",
    sample_level_fn=PassAtK(
        k=1,
        n=2,
        strip_strings=True,
        # Extracting mathematical expressions and latex expressions
        normalize_gold=lambda k: extract_target_from_pred(
            k,
            get_extraction_regexes(
                formatted_doc=None,
                target_types=[ExprExtractionConfig(), LatexExtractionConfig()],
                language=Language.ENGLISH,
            ),
        ),
        # Extracting mathematical expressions and latex expressions
        normalize_pred=lambda k: extract_target_from_pred(
            k,
            get_extraction_regexes(
                formatted_doc=None,
                target_types=[ExprExtractionConfig(), LatexExtractionConfig()],
                language=Language.ENGLISH,
            ),
        ),
        # Uses sympy for comparision
        sample_scoring_function=compare_gold_target,
    ).compute,
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)


def math_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["solution"]],
        gold_index=0,
    )


def aime_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def amc_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["answer"]],
        gold_index=0,
    )


def minerva_prompt_fn(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=MATH_QUERY_TEMPLATE.format(Question=line["problem"]),
        choices=[line["solution"]],
        gold_index=0,
    )


# Define tasks
aime24 = LightevalTaskConfig(
    name="aime24",
    suite=["custom"],
    prompt_function=aime_prompt_fn,
    hf_repo="HuggingFaceH4/aime_2024",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[
        # Metrics.math_pass_at_1_1n,
        # math_pass_at_1_2n,
        # Metrics.math_pass_at_1_4n,
        # Metrics.math_pass_at_1_16n,
        Metrics.math_pass_at_1_32n,
        # Metrics.math_pass_at_1_64n,
    ],
    version=1,
)

aime25 = LightevalTaskConfig(
    name="aime25",
    suite=["custom"],
    prompt_function=aime_prompt_fn,
    hf_repo="yentinglin/aime_2025",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[
        # Metrics.math_pass_at_1_1n,
        # math_pass_at_1_2n,
        # Metrics.math_pass_at_1_4n,
        # Metrics.math_pass_at_1_16n,
        Metrics.math_pass_at_1_32n,
        # Metrics.math_pass_at_1_64n,
    ],
    version=1,
)

amc23 = LightevalTaskConfig(
    name="amc23",
    suite=["custom"],
    prompt_function=amc_prompt_fn,
    hf_repo="knoveleng/AMC-23",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[
        # Metrics.math_pass_at_1_1n,
        # math_pass_at_1_2n,
        # Metrics.math_pass_at_1_4n,
        # Metrics.math_pass_at_1_16n,
        Metrics.math_pass_at_1_32n,
        # Metrics.math_pass_at_1_64n,
    ],
    version=1,
)

math_500 = LightevalTaskConfig(
    name="math_500",
    suite=["custom"],
    prompt_function=math_prompt_fn,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[
        # Metrics.math_pass_at_1_1n,
        math_pass_at_1_2n,
    ],
    version=1,
)

minerva = LightevalTaskConfig(
    name="minerva",
    suite=["custom"],
    prompt_function=minerva_prompt_fn,
    hf_repo="knoveleng/Minerva-Math",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=32768,
    metric=[
        # Metrics.math_pass_at_1_1n,
        # math_pass_at_1_2n,
        Metrics.math_pass_at_1_4n,
    ],
    version=1,
)


# Add tasks to the table
TASKS_TABLE = []
TASKS_TABLE.append(aime24)
TASKS_TABLE.append(aime25)
TASKS_TABLE.append(amc23)
TASKS_TABLE.append(math_500)
TASKS_TABLE.append(minerva)

# MODULE LOGIC
if __name__ == "__main__":
    print([t["name"] for t in TASKS_TABLE])
    print(len(TASKS_TABLE))

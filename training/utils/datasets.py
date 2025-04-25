from typing import Literal, Optional
from enum import Enum

import numpy as np
import verl.utils.torch_functional as verl_F
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from verl.utils.model import compute_position_id_with_mask

from reasoning_gym.coaching.experiment import Experiment
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.factory import get_score_answer_fn


class DatasetType(Enum):
    STATIC = "static" # static dataset stored on disk
    PROCEDURAL = "procedural" # procedural dataset generated on-the-fly
    EXPERIMENT = "experiment" # procedural dataset within an experiment


class ReasoningGymDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        static_dataset: list[dict] = None,
        procedural_dataset: Optional[ProceduralDataset] = None,
        experiment: Optional[Experiment] = None,
        developer_prompt: Optional[str] = None,
        developer_role: str = "system",
        max_prompt_length: int = 2048,
        truncation: str = "error",  ##  ['left', 'right', 'error']
    ):
        assert static_dataset or procedural_dataset or experiment, "One of `static_dataset`, `procedural_dataset` or `experiment` must be provided"
        assert sum(x is not None for x in [static_dataset, procedural_dataset, experiment]) == 1, "Exactly one of `static_dataset`, `procedural_dataset` or `experiment` must be provided"
        if static_dataset:
            self.dataset_type = DatasetType.STATIC
        if procedural_dataset:
            self.dataset_type = DatasetType.PROCEDURAL
        if experiment:
            self.dataset_type = DatasetType.EXPERIMENT

        self.data = static_dataset or procedural_dataset or experiment.composite
        self.experiment = experiment
        self.tokenizer = tokenizer
        self.developer_prompt = developer_prompt
        self.developer_role = developer_role
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        row_dict = self.data[index].copy()
        q = row_dict["question"]

        chat = []
        if self.developer_prompt is not None:
            chat.append({"role": self.developer_role, "content": self.developer_prompt})
        chat.append({"role": "user", "content": q})

        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["data_source"] = "reasoning_gym"
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["raw_prompt"] = chat
        row_dict["index"] = index
        return row_dict

    def score_answer(self, answer: str, index: int) -> float:
        """Score the answer using the underlying experiment's scorer."""
        entry = self.data[index]
        if self.dataset_type == DatasetType.EXPERIMENT:
            return self.experiment.score_answer_with_id(answer, entry["metadata"]["entry_id"])
        if self.dataset_type == DatasetType.PROCEDURAL:
            return self.data.score_answer(answer, entry=entry)
        if self.dataset_type == DatasetType.STATIC:
            score_fn = get_score_answer_fn(entry["metadata"]["source_dataset"])
            return score_fn(answer, entry)
        raise ValueError("No valid scoring method available")

    def update_experiment_difficulty(self, dataset_name: str, method: Literal["increment", "decrement"]):
        """Update the difficulty of the underlying dataset."""
        if self.experiment is None:
            raise ValueError("Cannot update difficulty: dataset is not a CurriculumExperiment")
        if method not in ["increment", "decrement"]:
            raise ValueError("Invalid method: must be 'increment' or 'decrement'")
        self.experiment.score_board.clear(dataset_name)
        self.experiment.update_difficulty(dataset_name, method)
        self.data = self.experiment.composite
        return True

    def aggregate(self, last_n: Optional[int] = None):
        """Aggregate scores from the underlying experiment"""
        if self.experiment is None:
            raise ValueError("Cannot aggregate scores: dataset is not a CurriculumExperiment")

        results = self.experiment.score_board.aggregate(last_n=last_n)
        output_results = {}

        for key, value in results.items():
            output_results[key] = {}
            scores = value.scores
            first_key = list(scores.keys())[0]
            output_results[key]["results"] = np.mean(scores[first_key])
            output_results[key]["total_samples"] = value.total_scores
        return output_results


def make_dataset(
    tokenizer,
    data_source: list[dict] | Experiment | ProceduralDataset,
    developer_prompt: str,
    max_prompt_length: int = 2048,
) -> ReasoningGymDataset:
    """
    Create ReasoningGymDataset object using either a ProceduralDataset or Experiment as the underlying data source.
    """
    if isinstance(data_source, Experiment):
        return ReasoningGymDataset(
            tokenizer=tokenizer,
            experiment=data_source,
            developer_prompt=developer_prompt,
            developer_role="system",
            max_prompt_length=max_prompt_length,
            truncation="error",
        )
    elif isinstance(data_source, ProceduralDataset):
        return ReasoningGymDataset(
            tokenizer=tokenizer,
            procedural_dataset=data_source,
            developer_prompt=developer_prompt,
            developer_role="system",
            max_prompt_length=max_prompt_length,
            truncation="error",
        )
    elif isinstance(data_source, list):
        return ReasoningGymDataset(
            tokenizer=tokenizer,
            static_dataset=data_source,
            developer_prompt=developer_prompt,
            developer_role="system",
            max_prompt_length=max_prompt_length,
            truncation="error",
        )
    else:
        raise ValueError("data_source must be either a Dataset, ProceduralDataset, or Experiment")
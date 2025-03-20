from typing import Literal, Optional
import numpy as np

import verl.utils.torch_functional as verl_F
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from verl.utils.model import compute_position_id_with_mask

from reasoning_gym.coaching.experiment import Experiment
from reasoning_gym.coaching.experiment import Experiment
from reasoning_gym.dataset import ProceduralDataset


class ReasoningGymDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        procedural_dataset: Optional[ProceduralDataset] = None,
        experiment: Optional[Experiment] = None,
        procedural_dataset: Optional[ProceduralDataset] = None,
        experiment: Optional[Experiment] = None,
        developer_prompt: Optional[str] = None,
        developer_role: str = "system",
        max_prompt_length: int = 2048,
        truncation: str = "error",  ##  ['left', 'right', 'error']
    ):
        assert procedural_dataset or experiment, "One of `procedural_dataset` or `experiment` must be provided"
        assert (
            procedural_dataset is None or experiment is None
        ), "Only one of `procedural_dataset` or `experiment` may be provided"

        assert procedural_dataset or experiment, "One of `procedural_dataset` or `experiment` must be provided"
        assert (
            procedural_dataset is None or experiment is None
        ), "Only one of `procedural_dataset` or `experiment` may be provided"

        self.tokenizer = tokenizer
        self.data = procedural_dataset or experiment.composite
        self.experiment = experiment
        self.data = procedural_dataset or experiment.composite
        self.experiment = experiment
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
        row_dict["data_source"] = "reasoning_gym"
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["raw_prompt"] = chat
        row_dict["index"] = index
        return row_dict
    
    def update_experiment_difficulty(self, dataset_name: str, method: Literal["increment", "decrement"]):
        """Update the difficulty of the underlying dataset."""  
        if self.experiment is None:
            raise ValueError("Cannot update difficulty: dataset is not a CurriculumExperiment")
        if method not in ["increment", "decrement"]:
            raise ValueError("Invalid method: must be 'increment' or 'decrement'")
        self.experiment.score_board.clear()
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
            assert len(scores.keys()) == 1, 'Maintain only one difficulty parameter at a time'
            first_key = list(scores.keys())[0]
            output_results[key]['results'] = np.mean(scores[first_key])
            output_results[key]['total_samples'] = value.total_scores
        
        return output_results

def make_dataset(
    tokenizer,
    data_source: Experiment | ProceduralDataset,
    dataset_name: str,
    developer_prompt: str,
) -> ReasoningGymDataset:
    """
    Create ReasoningGymDataset object using either a ProceduralDataset or Experiment as the underlying data source.
    """
    kwargs = {
        "tokenizer": tokenizer,
        #"dataset_name": dataset_name,
        "developer_prompt": developer_prompt,
    }
    if isinstance(data_source, Experiment):
        kwargs["experiment"] = data_source
    else:
        kwargs["procedural_dataset"] = data_source
    print(type(data_source))
    return ReasoningGymDataset(**kwargs)
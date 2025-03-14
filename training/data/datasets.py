from typing import Optional

import verl.utils.torch_functional as verl_F
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from verl.utils.model import compute_position_id_with_mask

import reasoning_gym
from reasoning_gym.composite import DatasetSpec


class ReasoningGymDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: str,
        seed: int,
        size: int,
        developer_prompt: Optional[str] = None,
        developer_role: str = "system",
        max_prompt_length: int = 2048,
        truncation: str = "error",  ##  ['left', 'right', 'error']
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.data = reasoning_gym.create_dataset(dataset_name, seed=seed, size=size)
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

        row_dict["data_source"] = "reasoning_gym/" + self.dataset_name
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["raw_prompt"] = chat.tolist()
        row_dict["index"] = index
        return row_dict


class ReasoningGymCompositeDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        datasets: list[dict],
        seed: int,
        size: int,
        developer_prompt: Optional[str] = None,
        developer_role: str = "system",
        max_prompt_length: int = 2048,
        truncation: str = "error",  ##  ['left', 'right', 'error']
    ):
        self.tokenizer = tokenizer
        self.data = reasoning_gym.create_dataset("composite", seed=seed, size=size)
        self.developer_prompt = developer_prompt
        self.developer_role = developer_role
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation

        for dataset in datasets:
            self.data.add_dataset(DatasetSpec(**dataset))

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

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

        # TODO: add data source names
        row_dict["data_source"] = "reasoning_gym/composite"
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["raw_prompt"] = chat.tolist()
        row_dict["index"] = index
        return row_dict

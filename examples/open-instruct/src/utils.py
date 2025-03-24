from typing import List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

import reasoning_gym
from reasoning_gym.utils import SYSTEM_PROMPTS


def list_preserving_collate(batch):
    """
    Custom collate function that preserves lists instead of converting to tensors.
    """
    token_ids = [item[0] for item in batch]
    items = [item[1] for item in batch]
    return token_ids, items


class ReasoningGymDataset(Dataset):
    def __init__(self, dataset_name, seed, size, tokenizer, developer_role, developer_prompt):
        self.data = reasoning_gym.create_dataset(dataset_name, seed=seed, size=size)
        self.tokenizer = tokenizer
        self.developer_role = developer_role
        self.developer_prompt = developer_prompt

    def __len__(self):
        return self.data.size

    def __getitem__(self, index):
        chat_message = [{"role": self.developer_role, "content": self.developer_prompt}]
        item = self.data[index]
        chat_message.append({"role": "user", "content": item["question"]})
        prompt_text = self.tokenizer.apply_chat_template(chat_message, tokenize=True, add_generation_prompt=True)
        prompt_text = [token for token in prompt_text if token != self.tokenizer.pad_token_id]
        return prompt_text, item


def pack_sequences(
    queries: List[List[int]],
    responses: List[List[int]],
    pack_length: int,
    pad_token_id: int,
) -> "PackedSequences":
    # assert padding token does not exist in queries and responses
    query_responses = []
    attention_masks = []
    response_masks = []
    num_actions = []
    packed_seq_lens = []
    cur_data = []
    cur_response_mask = []
    cur_num_actions = []
    cur_packed_seq_lens = []
    cur_attention_mask = []
    offset = 0

    for i in range(len(queries)):
        query = queries[i]
        response = responses[i]
        # remove padding (but using vllm so this should not be needed, but just in case)
        query = [t for t in query if t != pad_token_id]
        response = [t for t in response if t != pad_token_id]
        query_response = query + response

        if len(query_response) + len(cur_data) > pack_length:
            query_responses.append(cur_data)
            response_masks.append(cur_response_mask)
            attention_masks.append(cur_attention_mask)
            num_actions.append(cur_num_actions)
            packed_seq_lens.append(cur_packed_seq_lens)
            cur_data = []
            cur_response_mask = []
            cur_attention_mask = []
            cur_num_actions = []
            cur_packed_seq_lens = []
            offset = i

        cur_data.extend(query_response)
        cur_num_actions.append(len(response))
        cur_packed_seq_lens.append(len(query_response))
        cur_response_mask.extend([0 for _ in range(len(query))] + [i + 1 for _ in range(len(response))])
        cur_attention_mask.extend([i + 1 - offset for _ in range(len(query_response))])

    if len(cur_data) > 0:
        query_responses.append(cur_data)
        response_masks.append(cur_response_mask)
        attention_masks.append(cur_attention_mask)
        num_actions.append(cur_num_actions)
        packed_seq_lens.append(cur_packed_seq_lens)

    attention_masks_list = [torch.tensor(t) for t in attention_masks]

    from open_instruct.rl_utils2 import PackedSequences, reset_position_ids

    return PackedSequences(
        query_responses=[torch.tensor(t) for t in query_responses],
        attention_masks=attention_masks_list,
        position_ids=[reset_position_ids(t.unsqueeze(0)).squeeze(0) for t in attention_masks_list],
        response_masks=[torch.tensor(t) for t in response_masks],
        original_responses=responses,
        num_actions=[torch.tensor(t) for t in num_actions],
        packed_seq_lens=[torch.tensor(t) for t in packed_seq_lens],
    )

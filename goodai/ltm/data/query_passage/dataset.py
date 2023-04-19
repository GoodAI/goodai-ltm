import torch

from goodai.helpers.tokenizer_helper import get_model_inputs
from goodai.ltm.data.query_passage.data_source import BaseQueryPassageDataSource
from torch.utils.data import Dataset
from typing import List, Tuple
from transformers import PreTrainedTokenizer


class QueryPassageDataset(Dataset):
    def __init__(
        self,
        data_sources: List[Tuple[BaseQueryPassageDataSource, float]],
        tokenizer: PreTrainedTokenizer,
        num_examples: int,
        device: torch.device
    ):
        super().__init__()
        weight_sum = sum(w for _, w in data_sources)

        self.data_sources = data_sources
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.device = device
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise SystemError('Tokenizer has no PAD token.')

        label_list = []
        query_list = []
        passage_list = []

        for i, (ds, weight) in enumerate(data_sources):
            w = weight / weight_sum
            n = round(num_examples * w)
            items = ds.sample_items(n)
            for item in items:
                query_list.append(item.queryIds)
                passage_list.append(item.passageIds)
                label_list.append([1.0] if item.match else [0.0])

        query_inputs = get_model_inputs(query_list, pad_id, device, return_token_lengths=True, tokenizer=tokenizer)
        passage_inputs = get_model_inputs(passage_list, pad_id, device)
        self.query_input_ids = query_inputs['input_ids']
        self.query_att_mask = query_inputs['attention_mask']
        self.query_token_lengths = query_inputs['token_lengths']
        self.passage_input_ids = passage_inputs['input_ids']
        self.passage_att_mask = passage_inputs['attention_mask']
        self.labels = torch.as_tensor(label_list, dtype=torch.float, device=device)

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        q_input_ids = self.query_input_ids[idx]
        q_token_lengths = self.query_token_lengths[idx]
        q_att_mask = self.query_att_mask[idx]
        p_input_ids = self.passage_input_ids[idx]
        p_att_mask = self.passage_att_mask[idx]
        label = self.labels[idx]
        return (q_input_ids, q_token_lengths, q_att_mask), (p_input_ids, p_att_mask), label,

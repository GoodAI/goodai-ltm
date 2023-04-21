import itertools
import math
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer
from typing import List, Optional, Tuple

from goodai.helpers.tokenizer_helper import get_attention_after_token, get_model_inputs
from goodai.ltm.embedding_models.contrast_classifier import ContrastClassifier
from goodai.ltm.matching import BaseTextMatchingModel
from goodai.ltm.matching_models.prob_model import BaseQueryPassageProbModel


class DefaultRerankingCrossEncoder(nn.Module, BaseQueryPassageProbModel, BaseTextMatchingModel):
    sep_id_tensor: torch.Tensor

    def __init__(self, model_name: str, default_query_seq_len: Optional[int] = None,
                 default_passage_seq_len: Optional[int] = None, num_end_chars_lb_ignore=18):
        super().__init__()
        self.num_end_chars_lb_ignore = num_end_chars_lb_ignore
        self.default_passage_seq_len = default_passage_seq_len
        self.default_query_seq_len = default_query_seq_len
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size
        cls_emb_scale = math.sqrt(hidden_size)
        self.classifier = ContrastClassifier(cls_emb_scale)
        lb_token_ids = self.tokenizer.encode('\n', add_special_tokens=False)
        if len(lb_token_ids) != 1:
            raise ValueError(f'Tokenizer {self.tokenizer.name_or_path} does not have a line break token!')
        if self.tokenizer.sep_token_id is None:
            self.tokenizer.sep_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.sep_token_id is None:
            raise ValueError(f'Tokenizer {model_name} does not have SEP or EOS tokens!')
        if self.tokenizer.sep_token_id == self.tokenizer.pad_token_id:
            raise ValueError(f'Tokenizer {model_name} with SEP equal to PAD not supported!')
        self.lb_token_id = lb_token_ids[0]
        sep_id_tensor = torch.as_tensor([[self.tokenizer.sep_token_id]], dtype=torch.long)
        self.register_buffer('sep_id_tensor', sep_id_tensor)
        self.dummy = nn.Parameter()
        self.query_slfatt_model = nn.Sequential(
            nn.Dropout(p=0.03),
            nn.Linear(hidden_size, 1)
        )
        self.passage_slfatt_model = nn.Sequential(
            nn.Dropout(p=0.03),
            nn.Linear(hidden_size, 4)
        )

    def get_device(self):
        return self.dummy.device

    def get_lm_parameters(self):
        return self.model.parameters()

    def get_added_parameters(self):
        return itertools.chain(self.query_slfatt_model.parameters(),
                               self.passage_slfatt_model.parameters(),
                               self.classifier.parameters())

    @staticmethod
    def get_embedding(hidden_states: torch.Tensor, attention_mask: torch.Tensor,
                      att_model: nn.Module) -> torch.Tensor:
        # hidden_states: (batch_size, seq_len, emb_size,)
        x_att_mask = attention_mask[:, :, None]
        # x_att_mask: (batch_size, seq_len, 1,)
        slf_att_logits = att_model(hidden_states)
        # slf_att_logits: (batch_size, seq_len, total_keys,)
        slf_att_logits = slf_att_logits - 200.0 * (1.0 - x_att_mask)
        slf_att_logits = torch.clamp(slf_att_logits, min=-100, max=+80)
        slf_att_weights = torch.softmax(slf_att_logits, dim=1)
        # slf_att_weights: (batch_size, seq_len, total_keys,)
        dot_product = hidden_states[:, :, None, :] * slf_att_weights[:, :, :, None]
        raw_w_mean = torch.sum(dot_product, dim=1)
        # raw_w_mean: (batch_size, total_keys, emb_size,)
        return F.normalize(raw_w_mean, dim=-1)

    def forward(self, query_input_ids: torch.Tensor, query_attention_mask: torch.Tensor,
                query_token_lengths: torch.Tensor,
                passage_input_ids: torch.Tensor, passage_attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, query_len = query_input_ids.size(0), query_input_ids.size(1)
        passage_len = passage_input_ids.size(1)
        sep_t = torch.repeat_interleave(self.sep_id_tensor, batch_size, dim=0)
        input_ids = torch.hstack([query_input_ids, sep_t, passage_input_ids])
        attention_mask = torch.hstack([query_attention_mask, torch.ones_like(sep_t), passage_attention_mask])
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state
        query_hidden_states = last_hidden_states[:, :query_len]
        passage_hidden_states = last_hidden_states[:, -passage_len:]
        lb_att_mask = get_attention_after_token(query_input_ids, query_attention_mask, query_token_lengths,
                                                self.lb_token_id, exclude_last_n_chars=self.num_end_chars_lb_ignore,
                                                device=self.get_device())
        emb_q_att_mask = query_attention_mask * lb_att_mask
        query_embedding = self.get_embedding(query_hidden_states, emb_q_att_mask, self.query_slfatt_model)
        passage_embedding = self.get_embedding(passage_hidden_states, passage_attention_mask, self.passage_slfatt_model)
        match_probabilities = self.classifier(query_embedding, passage_embedding)
        return match_probabilities

    def get_probabilities_in_batches(self, query_input_ids: torch.Tensor, query_attention_mask: torch.Tensor,
                                     query_token_lengths: torch.Tensor,
                                     passage_input_ids: torch.Tensor, passage_attention_mask: torch.Tensor,
                                     max_batch_size: int, show_progress_bar: bool = False):
        batch_size = query_input_ids.size(0)
        output_probabilities_list = []
        rng = range(0, batch_size, max_batch_size)
        if show_progress_bar:
            rng = tqdm(rng, desc='Matching', unit='batch')
        for start_idx in rng:
            end_idx = start_idx + max_batch_size
            query_input_ids_batch = query_input_ids[start_idx:end_idx]
            query_attention_mask_batch = query_attention_mask[start_idx:end_idx]
            query_token_lengths_batch = query_token_lengths[start_idx:end_idx]
            passage_input_ids_batch = passage_input_ids[start_idx:end_idx]
            passage_attention_mask_batch = passage_attention_mask[start_idx:end_idx]
            output_probabilities_batch = self(query_input_ids_batch, query_attention_mask_batch,
                                              query_token_lengths_batch,
                                              passage_input_ids_batch, passage_attention_mask_batch)
            if not self.training:
                output_probabilities_batch = output_probabilities_batch.detach()
            output_probabilities_list.append(output_probabilities_batch)
        output_probabilities_t = torch.cat(output_probabilities_list)
        return output_probabilities_t

    def match_probabilities(self, sentences: List[Tuple[str, str]], fallback_max_query_tokens: int = 40,
                            fallback_max_passage_tokens: int = 70, show_progress_bar: bool = False,
                            max_batch_size: int = 50, use_preferred_seq_lengths: bool = True) -> List[float]:
        if len(sentences) == 0:
            return []
        max_query_tokens = None
        max_passage_tokens = None
        if use_preferred_seq_lengths:
            max_query_tokens = self.default_query_seq_len
            max_passage_tokens = self.default_passage_seq_len
        if max_query_tokens is None:
            max_query_tokens = fallback_max_query_tokens
        if max_passage_tokens is None:
            max_passage_tokens = fallback_max_passage_tokens
        tokenizer = self.tokenizer
        pad_id = tokenizer.pad_token_id
        query_ids_list = []
        passage_ids_list = []
        for query, passage in sentences:
            query_ids = tokenizer.encode(query, add_special_tokens=False)
            if len(query_ids) > max_query_tokens:
                query_ids = query_ids[-max_query_tokens:]
            passage_ids = tokenizer.encode(passage, add_special_tokens=False)
            if len(passage_ids) > max_passage_tokens:
                passage_ids = passage_ids[-max_passage_tokens:]
            passage_ids_list.append(passage_ids)
            query_ids_list.append(query_ids)
        # Convert input_ids_list to tensor input_ids and attention_mask by adding padding
        device = self.get_device()
        query_seq_len = self.default_query_seq_len if use_preferred_seq_lengths else None
        passage_seq_len = self.default_passage_seq_len if use_preferred_seq_lengths else None
        query_inputs = get_model_inputs(query_ids_list, pad_id, device, prefix='query_',
                                        min_seq_len=query_seq_len, return_token_lengths=True,
                                        tokenizer=tokenizer)
        passage_inputs = get_model_inputs(passage_ids_list, pad_id, device, prefix='passage_',
                                          min_seq_len=passage_seq_len)
        probabilities_t = self.get_probabilities_in_batches(**query_inputs, **passage_inputs,
                                                            max_batch_size=max_batch_size,
                                                            show_progress_bar=show_progress_bar)
        # probabilities_t: (batch_size, 1,)
        return probabilities_t.squeeze(1).tolist()

    def predict(self, sentences: List[Tuple[str, str]], batch_size: int = 32,
                show_progress_bar: bool = False) -> List[float]:
        return self.match_probabilities(sentences, max_batch_size=batch_size, show_progress_bar=show_progress_bar)

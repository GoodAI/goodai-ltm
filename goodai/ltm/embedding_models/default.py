import itertools
import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput

from goodai.helpers.tokenizer_helper import get_attention_after_token
from goodai.ltm.embedding_models.trainable import TrainableEmbeddingModel


class DefaultEmbeddingModel(TrainableEmbeddingModel):
    """
    Default implementation of trainable text embeddings.
    """

    def __init__(self, lang_model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                 num_retrieval_emb: int, num_storage_emb: int,
                 num_end_chars_lb_ignore=18, dropout=0.03):
        super(DefaultEmbeddingModel, self).__init__(tokenizer)
        lb_token_ids = tokenizer.encode('\n', add_special_tokens=False)
        valid_lb_token_id = len(lb_token_ids) == 1
        if not valid_lb_token_id:
            logging.warning(f'Tokenizer "{tokenizer.name_or_path}" does not have a line-break token.')
        self.lb_token_id = lb_token_ids[0] if valid_lb_token_id else -1
        self.num_end_chars_lb_ignore = num_end_chars_lb_ignore
        lm_config = lang_model.config
        self.lang_model = lang_model
        self.tokenizer = tokenizer
        self.num_storage_emb = num_storage_emb
        self.num_retrieval_emb = num_retrieval_emb
        self.emb_dim = self._get_embed_dim(lang_model, lang_model.config)
        hidden_size = lm_config.hidden_size
        out_size = num_retrieval_emb + num_storage_emb
        self.out_model = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, out_size),
        )
        self.dummy = nn.Parameter()

    @staticmethod
    def _get_embed_dim(model: PreTrainedModel, lm_config: PretrainedConfig):
        return lm_config.hidden_size

    def get_device(self):
        return self.dummy.device

    def forward(self, is_retrieve: bool, token_lengths: Optional[torch.Tensor], **kwargs):
        input_ids = kwargs['input_ids']
        attention_mask = kwargs.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if is_retrieve:
            if token_lengths is None:
                raise ValueError('token_lengths is mandatory when requesting retrieval keys')
            after_lb_att_mask = get_attention_after_token(input_ids, attention_mask, token_lengths, self.lb_token_id,
                                                          exclude_last_n_chars=self.num_end_chars_lb_ignore,
                                                          device=self.get_device())
            out_att_mask = attention_mask * after_lb_att_mask
        else:
            if token_lengths is not None:
                raise ValueError('token_lengths parameter was provided but is not expected for storage keys')
            out_att_mask = attention_mask
        lm_output: BaseModelOutput = self.lang_model(input_ids=input_ids,
                                                     attention_mask=attention_mask,
                                                     return_dict=True)
        token_embeddings = lm_output.last_hidden_state
        # token_embeddings: (batch_size, seq_len, emb_size,)
        slf_att_logits_1 = self.out_model(token_embeddings)
        slf_att_logits = slf_att_logits_1 - 200 * (1.0 - out_att_mask[:, :, None])
        slf_att_logits = torch.clamp(slf_att_logits, min=-100, max=+80)
        slf_att_weights = torch.softmax(slf_att_logits, dim=1)
        # slf_att_weights: (batch_size, seq_len, total_keys,)
        emb_product = token_embeddings[:, :, None, :] * slf_att_weights[:, :, :, None]
        # emb_product: (batch_size, seq_len, total_keys, emb_size,)
        if is_retrieve:
            relevant_product = emb_product[:, :, self.num_storage_emb:, :]
        else:
            relevant_product = emb_product[:, :, :self.num_storage_emb, :]
        raw_key_mean = torch.sum(relevant_product, dim=1)
        # raw_key_mean: (batch_size, num_keys, emb_size,)
        return F.normalize(raw_key_mean, dim=2)

    def get_lm_parameters(self):
        return self.lang_model.parameters()

    def get_added_parameters(self):
        return itertools.chain(self.out_model.parameters())

    def get_storage_emb(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self(input_ids=input_ids, token_lengths=None, attention_mask=attention_mask, is_retrieve=False)

    def get_retrieval_emb(self, input_ids: torch.Tensor, token_lengths: torch.Tensor,
                          attention_mask: torch.Tensor = None) -> torch.Tensor:
        return self(input_ids=input_ids, token_lengths=token_lengths, attention_mask=attention_mask,
                    is_retrieve=True)

    def get_emb(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sk = self.get_storage_emb(input_ids, attention_mask)
        rk = self.get_retrieval_emb(input_ids, attention_mask)
        return sk, rk,

    def get_embedding_dim(self) -> int:
        return self.emb_dim

    def get_num_retrieval_embeddings(self) -> int:
        return self.num_retrieval_emb

    def get_num_storage_embeddings(self) -> int:
        return self.num_storage_emb

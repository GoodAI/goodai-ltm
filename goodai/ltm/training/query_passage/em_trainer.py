import gc
import itertools
import math
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch import nn

from goodai.helpers.sched_opt import ScheduledOptimizer
from goodai.ltm.data.query_passage.data_source import BaseQueryPassageDataSource
from goodai.ltm.data.query_passage.auto_data_source import AutoQueryPassageDataSource
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
from timeit import default_timer as timer

from goodai.ltm.data.query_passage.dataset import QueryPassageDataset
from goodai.ltm.embeddings.trainable import TrainableEmbeddingModel
from goodai.modules.loss import EmbCrossProbLossModel


class EmbModelTrainer:
    def __init__(self, random: np.random.RandomState, tokenizer: PreTrainedTokenizer,
                 num_epochs: int, switch_ds_every: int, num_ds_examples: int, batch_size: int,
                 max_query_tokens: int, min_passage_tokens: int, max_passage_tokens: int,
                 track_validation: bool, lm_lr: float, extras_lr: float, device: torch.device,
                 num_warmup_steps: int = 0, weight_decay: float = 1e-3):
        super().__init__()
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.extras_lr = extras_lr
        self.lm_lr = lm_lr
        self.max_passage_tokens = max_passage_tokens
        self.min_passage_tokens = min_passage_tokens
        self.max_query_tokens = max_query_tokens
        self.track_validation = track_validation
        self.batch_size = batch_size
        self.num_ds_examples = num_ds_examples
        self.switch_ds_every = switch_ds_every
        self.num_epochs = num_epochs
        self.tokenizer = tokenizer
        self.random = random
        self.device = device
        self.train_data_sources: List[Tuple[BaseQueryPassageDataSource, float]] = []
        self.valid_data_sources: List[Tuple[BaseQueryPassageDataSource, float]] = []

    def add_data_source(self, ds_name: str, weight: float):
        train_ds, valid_ds = AutoQueryPassageDataSource.create(ds_name, self.random, self.tokenizer,
                                                               max_query_tokens=self.max_query_tokens,
                                                               min_passage_tokens=self.min_passage_tokens,
                                                               max_passage_tokens=self.max_passage_tokens)
        self.train_data_sources.append((train_ds, weight,))
        self.valid_data_sources.append((valid_ds, weight,))

    def train_dataset(self, dataset: Dataset, model: TrainableEmbeddingModel, loss_model: nn.Module,
                      s_opt: Optional[ScheduledOptimizer],
                      validation=False) -> Tuple[float, float]:
        training = not validation
        model.train(mode=training)
        loss_sum = 0
        bp_time = 0
        pass_count = 0
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for i in range(self.switch_ds_every):
            for q_tuple, p_tuple, _ in data_loader:
                qid, qtl, qam = q_tuple
                pid, pam = p_tuple
                a_embeddings = model(input_ids=qid, attention_mask=qam, token_lengths=qtl, is_retrieve=True)
                b_embeddings = model(input_ids=pid, attention_mask=pam, token_lengths=None, is_retrieve=False)
                loss = loss_model(a_embeddings, b_embeddings)
                loss_sum += loss.item()
                if training:
                    s_opt.zero_grad()
                    time1 = timer()
                    gc.collect()
                    loss.backward()
                    time2 = timer()
                    bp_time += time2 - time1
                    s_opt.step()
                    s_opt.zero_grad(set_to_none=True)
                del a_embeddings
                del b_embeddings
                del loss
                gc.collect()
                pass_count += 1
        return loss_sum / pass_count, bp_time,

    def train(self, model: TrainableEmbeddingModel):
        print('------------------------------------')
        print(f'ML learning rate: {self.lm_lr}')
        print(f'Extras learning rate: {self.extras_lr}')
        print(f'Batch size: {self.batch_size}')
        print(f'Num epochs per dataset: {self.switch_ds_every}')
        print(f'Num epochs: {self.num_epochs}')
        print(f'Max query tokens: {self.max_query_tokens}')
        print(f'Min passage tokens: {self.min_passage_tokens}')
        print(f'Max passage tokens: {self.max_passage_tokens}')
        print('------------------------------------')
        if self.num_epochs > 0:
            print('Training...')
            loss_model = EmbCrossProbLossModel().to(self.device)
            lm_parameters = model.get_lm_parameters()
            extra_parameters = model.get_added_parameters()
            all_extra_parameters = itertools.chain(extra_parameters, loss_model.parameters())
            num_training_steps = self.num_epochs * math.ceil(self.num_ds_examples / self.batch_size)
            parameters = [
                {'params': lm_parameters, 'lr': self.lm_lr},
                {'params': all_extra_parameters, 'lr': self.extras_lr},
            ]
            s_opt = ScheduledOptimizer(parameters, lr=self.lm_lr,
                                       num_training_steps=num_training_steps,
                                       num_warmup_steps=self.num_warmup_steps,
                                       weight_decay=self.weight_decay)
            sum_bp_time = 0
            time1 = timer()
            for epoch in range(0, self.num_epochs, self.switch_ds_every):
                dataset = self.create_dataset(training=True)
                train_loss, bp_time = self.train_dataset(dataset, model, loss_model, s_opt, validation=False)
                sum_bp_time += bp_time
                print(f'-- Epoch {epoch}: Train Loss={train_loss:.4g}')
                last_lrs = s_opt.get_last_lrs()
                print(f'Last LM LR: {np.round(last_lrs, 6)}')
                if self.track_validation and epoch % 20 == 0:
                    valid_ds = self.create_dataset(training=False)
                    valid_loss, _ = self.train_dataset(valid_ds, model, loss_model, None,
                                                       validation=True)
                    print(f'== Epoch {epoch}: Valid Loss={valid_loss:.4g}')
            time2 = timer()
            elapsed = time2 - time1
            print(f'Trained {self.num_epochs} epochs in {elapsed:.1f} seconds.')
            bp_time_pct = sum_bp_time * 100.0 / elapsed
            print(f'Backprop time: {sum_bp_time:.1f} seconds ({bp_time_pct:.1f}%)')
            eph = self.num_epochs / (elapsed / (60 * 60))
            print(f'Training speed: {eph:.1f} epochs per hour.')
            torch.cuda.empty_cache()
            gc.collect()

    def create_dataset(self, training: bool):
        data_sources = self.train_data_sources if training else self.valid_data_sources
        actual_ds_examples = self.num_ds_examples
        return QueryPassageDataset(data_sources, self.tokenizer, actual_ds_examples,
                                   device=self.device, approx_positive_fraction=1.0)

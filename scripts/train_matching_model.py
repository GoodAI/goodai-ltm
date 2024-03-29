import gc
import logging
import os
import pickle
import uuid
from typing import Optional

import click
import numpy as np
import torch
from dotenv import load_dotenv

from goodai.helpers.torch_helper import param_count
from goodai.ltm.data.cloud import CloudStorage
from goodai.ltm.reranking.default import DefaultRerankingCrossEncoder

from transformers import AutoTokenizer

from goodai.ltm.training.query_passage.qppm_trainer import QPPMTrainer

load_dotenv()
_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
_data_sources = [
    ('squad_v2', 100.0),
    ('coqa', 40.0),
]


@click.command()
@click.option('--model-name', default='sentence-transformers/all-distilroberta-v1', type=str,
              help='Huggingface model that will be fine-tuned')
@click.option('--batch-size', default=200, type=int, help='Batch size')
@click.option('--num-epochs', default=200, type=int, help='The number of epochs')
@click.option('--save', default=False, type=bool, help='Whether to save the model in S3 bucket', is_flag=True)
@click.option('--lm-lr', default=1.5e-6, type=float, help='Learning rate of parameters of pretrained language model')
@click.option('--extras-lr', default=5e-5, type=float, help='Learning rate of parameters added to model')
@click.option('--seed', default=7002, type=int, help='Randomization seed')
@click.option('--switch-ds-every', default=1, type=int, help='How often (epochs) to switch generated dataset')
@click.option('--num-ds-examples', default=600, type=int, help='Number of examples in generated dataset')
@click.option('--max-query-tokens', default=40, type=int, help='Maximum number of query tokens')
@click.option('--min-passage-tokens', default=24, type=int, help='Minimum number of passage tokens')
@click.option('--max-passage-tokens', default=36, type=int, help='Maximum number of passage tokens')
def run(model_name: str, batch_size: int, num_epochs: int,
        save: bool, lm_lr: float, extras_lr: float, seed: int, switch_ds_every: int, num_ds_examples: int,
        max_query_tokens: int, min_passage_tokens: int, max_passage_tokens: int,
        track_validation: bool = True, eval_limit: Optional[int] = None, autograd_anomaly_detect: bool = False):
    if autograd_anomaly_detect:
        logging.warning('Enabling autograd anomaly detection.')
        torch.autograd.set_detect_anomaly(autograd_anomaly_detect)
    if not torch.cuda.is_available():
        logging.warning('CUDA not available.')
    print(f'Device: {_device}')
    print(f'Loading model "{model_name}"...')
    if not save:
        logging.warning('Trained model will not be saved.')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DefaultRerankingCrossEncoder(model_name, default_query_seq_len=max_query_tokens,
                                         default_passage_seq_len=max_passage_tokens)
    model = model.to(_device)
    num_params = param_count(model)
    print(f'Num model params: {num_params / 1e+6:.1f} million.')
    random = np.random.RandomState(seed)
    print(f'Will train for {num_epochs} epochs with LR {lm_lr:.4g} [{extras_lr:.4g}] and batch size {batch_size}')
    trainer = QPPMTrainer(random, tokenizer, num_epochs, switch_ds_every, num_ds_examples,
                          batch_size, max_query_tokens, min_passage_tokens, max_passage_tokens,
                          track_validation, lm_lr, extras_lr, _device)
    for ds_name, weight in _data_sources:
        trainer.add_data_source(ds_name, weight)
    trainer.train(model)
    model.eval()
    # TODO evaluation
    # vector_db = SimpleVectorDb()
    # memory = DefaultTextMemory(vector_db, tokenizer, emb_model, model, _device, TextMemoryConfig())
    if save:
        del trainer
        gc.collect()
        model = model.cpu()
        model_object = dict(
            base_model_name=model_name,
            model=model,
        )
        bucket_name = os.environ.get('AWS_BUCKET')
        if bucket_name is None:
            raise RuntimeError('AWS_BUCKET needs to be set to save models')
        print('Saving model...')
        storage = CloudStorage.get_instance(bucket_name)
        model_id = uuid.uuid4().hex
        model_key = f'matching-model-{model_id}'
        model_bytes = pickle.dumps(model_object)
        storage.put_object_bytes(model_key, model_bytes)
        print(f'Saved model {model_key}')
    else:
        print('Not saving model.')


if __name__ == '__main__':
    run()

from dataclasses import dataclass
from typing import Optional, List

import torch
from transformers import AutoTokenizer

from goodai.ltm.embedding_models.auto import AutoTextEmbeddingModel
from goodai.ltm.eval.auto import AutoMemEvaluator
from goodai.ltm.matching_models.auto import AutoTextMatchingModel
from goodai.ltm.memory_models.config import TextMemoryConfig
from goodai.ltm.memory_models.default import DefaultTextMemory
from goodai.ltm.memory_models.simple_vector_db import SimpleVectorDb


@dataclass
class EvalSpec:
    id: str
    embModelName: str
    matchingModelName: Optional[str]
    maxQueryTokens: int
    hasQueryNoise: bool


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    top_ks = [3, 10]
    vector_db = SimpleVectorDb()
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
    datasets = ['qrecc']
    # datasets = ['qrecc', 'strategyqa']
    eval_specs: List[EvalSpec] = [
        EvalSpec('all-distilroberta-v1', 'st:sentence-transformers/all-distilroberta-v1', None,
                 maxQueryTokens=40, hasQueryNoise=False),
    ]
    ds_top_ks = [f'{ds_name}@{top_k}' for ds_name in datasets for top_k in top_ks]
    table_out = 'Model | ' + ' | '.join([ds_name for ds_name in ds_top_ks]) + '\n'
    table_out += '----- | ' + ' | '.join(['-' * len(ds_name) for ds_name in ds_top_ks]) + '\n'
    for spec in eval_specs:
        max_query_tokens = spec.maxQueryTokens
        has_query_noise = spec.hasQueryNoise
        emb_model = AutoTextEmbeddingModel.from_pretrained(spec.embModelName)
        mmn = spec.matchingModelName
        matching_model = None if mmn is None else AutoTextMatchingModel.from_pretrained(mmn)
        config = TextMemoryConfig()
        mem = DefaultTextMemory(vector_db, tokenizer, emb_model, matching_model,
                                device=device, config=config)
        table_out += spec.id + ' | '
        for dataset in datasets:
            print(f'Evaluation of {spec.id} on {dataset}...')
            evaluator = AutoMemEvaluator.create(dataset, tokenizer, top_ks, max_query_tokens, has_query_noise)
            results = evaluator.evaluate(mem)
            for top_k in top_ks:
                acc = results[f'ACC@{top_k}']
                acc_100 = f'{acc * 100:.02f}'
                table_out += acc_100 + ' | '
        table_out += '\n'
    print('Results:')
    print(table_out)
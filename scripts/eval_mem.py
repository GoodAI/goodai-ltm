from dataclasses import dataclass
from typing import Optional, List
import torch
from transformers import AutoTokenizer
from goodai.ltm.embeddings.auto import AutoTextEmbeddingModel
from goodai.ltm.eval.auto import AutoMemEvaluator
from goodai.ltm.mem.mem_foundation import VectorDbType
from goodai.ltm.reranking.auto import AutoTextMatchingModel
from goodai.ltm.mem.config import TextMemoryConfig
from goodai.ltm.mem.default import DefaultTextMemory


@dataclass
class EvalSpec:
    id: str
    embModelName: str
    matchingModelName: Optional[str] = None
    maxQueryTokens: int = 40
    hasQueryNoise: bool = True
    chunkCapacity: int = 24
    rerankingKFactor: int = 10

    @classmethod
    def for_qpm(cls, qpm_model_name: str, emb_model_name: str, reranking_k_factor: int):
        m_id = f'x{reranking_k_factor} w/ {emb_model_name}'
        return cls(id=m_id, embModelName=emb_model_name, matchingModelName=qpm_model_name,
                   rerankingKFactor=reranking_k_factor)


_hf_eval_specs_0 = [EvalSpec(mid, mid) for mid in [
    'st:sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
]]

_hf_eval_specs_1 = [EvalSpec(mid, mid) for mid in [
    'st:sentence-transformers/all-distilroberta-v1',
    'st:sentence-transformers/sentence-t5-large',
]]

_hf_eval_specs_2 = [EvalSpec(mid, mid) for mid in [
    'st:sentence-transformers/multi-qa-mpnet-base-cos-v1',
    'st:sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
    'st:sentence-transformers/all-mpnet-base-v2',
]]

_hf_eval_specs_3 = [EvalSpec(mid, mid) for mid in [
    'st:sentence-transformers/all-roberta-large-v1',
    'st:sentence-transformers/sentence-t5-xxl',
]]

_goodai_eval_specs = [EvalSpec(mid, mid) for mid in [
    'em-distilroberta-p1-01',
    'em-distilroberta-p3-01',
    'em-MiniLM-p3-01',
]]

_goodai_eval_specs_2 = [EvalSpec(mid, mid) for mid in [
    'em-MiniLM-p1-01',
    'em-distilroberta-p5-01',
]]

_openai_eval_specs = [EvalSpec(mid, mid) for mid in [
    'openai:text-embedding-ada-002',
]]

_qpm_eval_specs_1 = [EvalSpec.for_qpm('qpm-distilroberta-01', mid, rkf) for mid, rkf in [
    ('em-distilroberta-p3-01', 10),
    ('em-distilroberta-p3-01', 5),
]]

_qpm_eval_specs_2 = [EvalSpec.for_qpm('qpm-distilroberta-01', mid, rkf) for mid, rkf in [
    ('st:sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 2),
    ('st:sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 3),
]]

_qpm_eval_specs_3 = [EvalSpec.for_qpm('qpm-distilroberta-01', mid, rkf) for mid, rkf in [
    ('em-distilroberta-p3-01', 2),
    ('em-distilroberta-p3-01', 3),
]]

_qpm_eval_specs_4 = [EvalSpec.for_qpm('qpm-distilroberta-01', mid, rkf) for mid, rkf in [
    ('st:sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 4),
    ('st:sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 5),
]]

_qpm_eval_specs_5 = [EvalSpec.for_qpm('qpm-distilroberta-01', mid, rkf) for mid, rkf in [
    ('em-distilroberta-p3-01', 4),
    ('em-distilroberta-p3-01', 5),
]]

_qpm_eval_specs_6 = [EvalSpec.for_qpm('qpm-distilroberta-01', mid, rkf) for mid, rkf in [
    ('em-distilroberta-p1-01', 2),
    ('em-distilroberta-p1-01', 3),
]]

_qpm_eval_specs_7 = [EvalSpec.for_qpm('qpm-distilroberta-01', mid, rkf) for mid, rkf in [
    ('st:sentence-transformers/multi-qa-mpnet-base-cos-v1', 2),
    ('st:sentence-transformers/multi-qa-mpnet-base-cos-v1', 3),
]]

_qpm_eval_specs_8 = [EvalSpec.for_qpm('qpm-distilroberta-01', mid, rkf) for mid, rkf in [
    ('st:sentence-transformers/all-distilroberta-v1', 2),
    ('st:sentence-transformers/all-distilroberta-v1', 3),
]]

_qpm_eval_specs_9 = [EvalSpec.for_qpm('qpm-distilroberta-01', mid, rkf) for mid, rkf in [
    ('st:sentence-transformers/multi-qa-mpnet-base-cos-v1', 4),
    ('st:sentence-transformers/multi-qa-mpnet-base-cos-v1', 5),
]]

_qpm_eval_specs_10 = [EvalSpec.for_qpm('qpm-distilroberta-01', mid, rkf) for mid, rkf in [
    ('st:sentence-transformers/multi-qa-mpnet-base-cos-v1', 10),
    ('st:sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 10),
]]

_qpm_eval_specs_11 = [EvalSpec.for_qpm('qpm-distilroberta-01', mid, rkf) for mid, rkf in [
    ('st:sentence-transformers/multi-qa-mpnet-base-cos-v1', 7),
    ('st:sentence-transformers/multi-qa-MiniLM-L6-cos-v1', 7),
]]

_qpm_eval_specs_12 = [EvalSpec.for_qpm('em:em-distilroberta-p5-01', mid, rkf) for mid, rkf in [
    ('em-MiniLM-p1-01', 2),
    ('em-MiniLM-p1-01', 3),
]]

_qpm_eval_specs_13 = [EvalSpec.for_qpm('em:em-distilroberta-p5-01', mid, rkf) for mid, rkf in [
    ('em-MiniLM-p1-01', 5),
    ('em-MiniLM-p1-01', 10),
]]

_qpm_eval_specs_14 = [EvalSpec.for_qpm('em:em-distilroberta-p5-01', mid, rkf) for mid, rkf in [
    ('em-MiniLM-p1-01', 15),
    ('em-MiniLM-p1-01', 20),
]]

_qpm_eval_specs_15 = [EvalSpec.for_qpm('em:em-distilroberta-p5-01', mid, rkf) for mid, rkf in [
    ('em-MiniLM-p1-01', 1),
    ('em-MiniLM-p1-01', 8),
]]

_qpm_eval_specs_17 = [EvalSpec.for_qpm('em:em-distilroberta-p5-01', mid, rkf) for mid, rkf in [
    ('em-distilroberta-p5-01', 1),
]]

if __name__ == '__main__':
    eval_specs: List[EvalSpec] = _qpm_eval_specs_14

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(1001)
    top_ks = [3, 10]
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    datasets = ['qrecc', 'strategyqa', 'msmarco']
    # datasets = ['qrecc']
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
        config.chunk_capacity = spec.chunkCapacity
        config.reranking_k_factor = spec.rerankingKFactor
        # Query truncation taken care of by the evaluator class
        config.max_query_length = None
        table_out += spec.id + ' | '
        for dataset in datasets:
            print(f'Evaluation of {spec.id} on {dataset}...')
            # Each dataset-model needs its own memory object
            mem = DefaultTextMemory(VectorDbType.SIMPLE, tokenizer, emb_model, matching_model,
                                    device=device, config=config)
            evaluator = AutoMemEvaluator.create(dataset, tokenizer, top_ks, max_query_tokens, has_query_noise)
            results = evaluator.evaluate(mem)
            for top_k in top_ks:
                acc = results[f'ACC@{top_k}']
                acc_100 = f'{acc * 100:.02f}'
                table_out += acc_100 + ' | '
        table_out += '\n'
    print('Results:')
    print(table_out)

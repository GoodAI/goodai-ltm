from typing import List

from transformers import PreTrainedTokenizer

from goodai.ltm.eval.qrecc import QreccMemEvaluator
from goodai.ltm.eval.strategyqa import StrategyQAMemEvaluator


class AutoMemEvaluator:
    @staticmethod
    def create(name: str, tokenizer: PreTrainedTokenizer, top_ks: List[int], max_query_tokens: int,
               has_query_noise: bool):
        if name == 'qrecc':
            return QreccMemEvaluator(tokenizer, top_ks, max_query_tokens, has_query_noise)
        elif name == 'strategyqa':
            return StrategyQAMemEvaluator(tokenizer, top_ks, max_query_tokens, has_query_noise)
        else:
            raise ValueError(f'Dataset name "{name}" not recognized')

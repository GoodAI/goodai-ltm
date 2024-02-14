import codecs
import json
import os
from typing import List, Optional

import datasets
import numpy as np
from transformers import PreTrainedTokenizer

from goodai.helpers.file_helper import download_zip
from goodai.ltm.eval.mem import BaseMemEvaluator, QAScenario


class MsMarcoMemEvaluator(BaseMemEvaluator):
    def __init__(self, tokenizer: PreTrainedTokenizer, top_ks: List[int],
                 max_query_tokens: int, has_query_noise: bool, use_rewrite: bool = True,
                 max_scenarios: int = 2000):
        super().__init__(tokenizer, top_ks, max_query_tokens, has_query_noise)
        self.use_rewrite = use_rewrite
        ds = datasets.load_dataset('ms_marco', 'v2.1', ignore_verifications=True)
        valid_data = ds['validation']
        self.data = valid_data[:max_scenarios]

    def get_facts_to_be_stored(self) -> List[str]:
        facts = []
        d = self.data
        answers, passages, query = d['answers'], d['passages'], d['query']
        for a, p, q in zip(answers, passages, query):
            is_selected = p['is_selected']
            if np.any(is_selected):
                a_len = [len(x) for x in a]
                a_idx = np.argmax(a_len)
                facts.append(a[a_idx])
        return facts

    @staticmethod
    def get_context(passages: dict) -> Optional[List[str]]:
        is_selected = passages['is_selected']
        if not np.any(is_selected):
            return None
        passage_text = passages['passage_text']
        return [p for p, s in zip(passage_text, is_selected) if not s]

    def get_scenarios(self) -> List[QAScenario]:
        scenarios = []
        d = self.data
        answers, passages, query = d['answers'], d['passages'], d['query']
        for a, p, q in zip(answers, passages, query):
            context = self.get_context(p)
            if context is not None:
                a_len = [len(x) for x in a]
                a_idx = np.argmax(a_len)
                sf = [a[a_idx]]
                scenario = QAScenario(context=context, supportingFacts=sf, question=q)
                scenarios.append(scenario)
        return scenarios

import codecs
import json
import os
from typing import List, Optional

import datasets
import numpy as np
from transformers import PreTrainedTokenizer

from goodai.helpers.file_helper import download_zip
from goodai.ltm.data.query_passage.auto_data_source import AutoQueryPassageDataSource
from goodai.ltm.eval.mem import BaseMemEvaluator, QAScenario


class QPDSMemEvaluator(BaseMemEvaluator):
    # For testing query-passage data sources
    def __init__(self, ds_name: str, tokenizer: PreTrainedTokenizer, top_ks: List[int],
                 max_query_tokens: int, has_query_noise: bool, use_rewrite: bool = True,
                 max_scenarios: int = 2000):
        super().__init__(tokenizer, top_ks, max_query_tokens, has_query_noise, add_names_to_context=False)
        self.use_rewrite = use_rewrite
        rnd = np.random.RandomState(7089)
        _, ds = AutoQueryPassageDataSource.create(ds_name, rnd, tokenizer, max_query_tokens=max_query_tokens)
        examples = ds.sample_items(max_scenarios, approx_positive_fraction=1.0)
        entries = []
        for ex in examples:
            query = tokenizer.decode(ex.queryIds, skip_special_tokens=True)
            passage = tokenizer.decode(ex.passageIds, skip_special_tokens=True)
            entries.append((query, passage,))
        self.entries = entries

    def get_facts_to_be_stored(self) -> List[str]:
        return [p for _, p in self.entries]

    def get_scenarios(self) -> List[QAScenario]:
        scenarios = []
        for q, p in self.entries:
            scenario = QAScenario(context=[], supportingFacts=[p], question=q)
            scenarios.append(scenario)
        return scenarios

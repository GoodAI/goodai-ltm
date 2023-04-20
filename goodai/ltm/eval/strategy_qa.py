import codecs
import json
import os
from typing import List

import numpy as np
from transformers import PreTrainedTokenizer

from goodai.helpers.file_helper import download_zip
from goodai.ltm.eval.mem import BaseMemEvaluator, QAScenario

_url = 'https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip'
_test_file = 'strategyqa_train.json'


class StrategyQAMemEvaluator(BaseMemEvaluator):
    def __init__(self, tokenizer: PreTrainedTokenizer, top_ks: List[int],
                 max_query_tokens: int, has_query_noise: bool, use_rewrite: bool = True,
                 max_scenarios: int = 2000, max_num_facts_per_scenario: int = 1):
        super().__init__(tokenizer, top_ks, max_query_tokens, has_query_noise)
        self.max_num_facts_per_scenario = max_num_facts_per_scenario
        self.use_rewrite = use_rewrite
        data_dir = download_zip(_url)
        data_file = os.path.join(data_dir, _test_file)
        with codecs.open(data_file, 'r', 'utf-8') as fd:
            all_data = json.load(fd)
            rnd = np.random.RandomState(8011)
            rnd.shuffle(all_data)
            self.data = all_data[:max_scenarios]

    def get_facts_to_be_stored(self) -> List[str]:
        facts = []
        for entry in self.data:
            entry_facts = entry.get('facts')
            if entry_facts:
                facts.extend(entry_facts[:self.max_num_facts_per_scenario])
        return facts

    def get_scenarios(self) -> List[QAScenario]:
        scenarios = []
        for entry in self.data:
            entry_facts = entry.get('facts')
            if entry_facts:
                description = entry.get('description', '')
                q = entry.get('question')
                if q:
                    context = [description]
                    s_facts = entry_facts[:self.max_num_facts_per_scenario]
                    scenario = QAScenario(context=context, supportingFacts=s_facts, question=q)
                    scenarios.append(scenario)
        return scenarios

import codecs
import json
import os
from typing import List
from transformers import PreTrainedTokenizer

from goodai.helpers.file_helper import download_zip
from goodai.ltm.eval.mem import BaseMemEvaluator, QAScenario

_url = 'https://github.com/apple/ml-qrecc/blob/main/dataset/qrecc_data.zip?raw=true'
_test_file = 'qrecc_test.json'


class QreccMemEvaluator(BaseMemEvaluator):
    def __init__(self, tokenizer: PreTrainedTokenizer, top_ks: List[int],
                 max_query_tokens: int, has_query_noise: bool, use_rewrite: bool = True):
        super().__init__(tokenizer, top_ks, max_query_tokens, has_query_noise)
        self.use_rewrite = use_rewrite
        data_dir = download_zip(_url)
        data_file = os.path.join(data_dir, _test_file)
        with codecs.open(data_file, 'r', 'utf-8') as fd:
            self.data = json.load(fd)

    def get_facts_to_be_stored(self) -> List[str]:
        facts = []
        for entry in self.data:
            answer = entry.get('Answer')
            if answer:
                facts.append(answer)
        return facts

    def get_scenarios(self) -> List[QAScenario]:
        scenarios = []
        for entry in self.data:
            answer = entry.get('Answer')
            if answer:
                context = entry.get('Context')
                if context is not None:
                    q = entry.get('Rewrite') if self.use_rewrite else entry.get('Question')
                    if q:
                        sf = [answer]
                        scenario = QAScenario(context=context, supportingFacts=sf, question=q)
                        scenarios.append(scenario)
        return scenarios

import abc
from abc import ABC
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from goodai.ltm.data.names import NameSource
from goodai.ltm.eval.metrics import get_correctness_score
from goodai.ltm.memory import BaseTextMemory


@dataclass
class QAScenario:
    context: List[str]
    supportingFacts: List[str]
    question: str


class BaseMemEvaluator(ABC):
    def __init__(self, tokenizer: PreTrainedTokenizer, top_ks: List[int],
                 max_query_tokens: int, has_query_noise: bool,
                 add_names_to_context: bool = True, correctness_threshold=70):
        if len(top_ks) == 0:
            raise ValueError('At least one top-k must be provided')
        self.top_ks = top_ks
        self.correctness_threshold = correctness_threshold
        self.tokenizer = tokenizer
        self.max_query_tokens = max_query_tokens
        self.has_query_noise = has_query_noise
        self.add_names_to_context = add_names_to_context
        self.rnd1 = np.random.RandomState(8001)
        self.rnd2 = np.random.RandomState(8002)

    @abc.abstractmethod
    def get_facts_to_be_stored(self) -> List[str]:
        pass

    @abc.abstractmethod
    def get_scenarios(self) -> List[QAScenario]:
        pass

    def get_query(self, scenario: QAScenario) -> str:
        name1, name2 = tuple(NameSource.get_instance().sample_first_names(self.rnd1, count=2))
        names = [name1, name2]
        names_context = [f'{names[i % 2]}: {ctx}' for i, ctx in enumerate(scenario.context)]
        c_len = len(names_context)
        q_name = names[c_len % 2]
        a_name = names[(c_len + 1) % 2]
        query = f'{q_name}: {scenario.question}\n{a_name}:'
        if self.has_query_noise:
            context_as_text = '\n'.join(names_context[-3:])
            query = context_as_text + '\n' + query
        plain_query_token_ids = self.tokenizer.encode(query, add_special_tokens=False)
        plain_query_token_ids = plain_query_token_ids[-self.max_query_tokens:]
        return self.tokenizer.decode(plain_query_token_ids, skip_special_tokens=True)

    def get_queries_and_support(self, scenarios: List[QAScenario]) -> Tuple[List[str], List[List[str]]]:
        queries = []
        supports_list: List[List[str]] = []
        for scenario in scenarios:
            query = self.get_query(scenario)
            support = scenario.supportingFacts
            queries.append(query)
            supports_list.append(support)
        return queries, supports_list,

    def cross_max_correctness(self, retrieved_texts: List[str], supporting_facts: List[str]):
        result = []
        for rt in retrieved_texts:
            scores = [get_correctness_score(self.tokenizer, rt, f) for f in supporting_facts]
            result.append(max(scores))
        return result

    def evaluate(self, memory: BaseTextMemory) -> Dict[str, float]:
        facts = self.get_facts_to_be_stored()
        for fact in tqdm(facts, desc='Storage', unit='fact'):
            memory.add_text(fact + '\n')
        scenarios = self.get_scenarios()
        queries, supports = self.get_queries_and_support(scenarios)
        k = max(self.top_ks)
        retrieved = memory.retrieve_multiple(queries, k=k, show_progress_bar=True)
        top_k_map: Dict[int, int] = dict()
        for s_retrieved, s_supports in tqdm(zip(retrieved, supports), desc='Comparison', unit='scenario'):
            s_retrieved_texts = [r.passage for r in s_retrieved]
            correctness_values = self.cross_max_correctness(s_retrieved_texts, s_supports)
            for top_k in self.top_ks:
                selected_cv = correctness_values[:top_k]
                if max(selected_cv) >= self.correctness_threshold:
                    top_k_map[top_k] = top_k_map.get(top_k, 0) + 1
        item_count = len(retrieved)
        return {f'ACC@{top_k}': v / item_count for top_k, v in top_k_map.items()}

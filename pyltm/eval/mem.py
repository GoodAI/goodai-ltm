import abc
from abc import ABC
from typing import List
from tqdm import tqdm
from pyltm.memory import BaseTextMemory


class QAScenario:
    context: List[str]
    supportingFacts: List[str]
    question: str


class BaseMemEvaluator(ABC):
    def __init__(self, top_ks: List[int]):
        if len(top_ks) == 0:
            raise ValueError('At least one top-k must be provided')
        self.top_ks = top_ks

    @abc.abstractmethod
    def get_facts(self) -> List[str]:
        pass

    @abc.abstractmethod
    def get_scenarios(self) -> List[QAScenario]:
        pass

    def evaluate(self, memory: BaseTextMemory):
        facts = self.get_facts()
        for fact in tqdm(facts, desc='Storage', unit='fact'):
            memory.add_text(fact + '\n')
        scenarios = self.get_scenarios()
        queries, supports = self.get_queries_and_support(scenarios)
        k = max(self.top_ks)
        retrieved = memory.retrieve_multiple(queries, k=k)


import enum
import logging
import math
import time
from typing import List
from goodai.ltm.mem.base import BaseReranker, RetrievedMemory, BaseTextMemory


class DecayType(enum.Enum):
    EXPONENTIAL = 0,
    INVERSE = 1,


class StanfordReranker(BaseReranker):
    def __init__(self, half_life: float,
                 multiplicative: bool = False,
                 decay_type: DecayType = DecayType.EXPONENTIAL,
                 alpha_recency: float = 1.0, alpha_importance: float = 1.0,
                 alpha_relevance: float = 1.0):
        """
        Constructs a StanfordReranker, per https://arxiv.org/pdf/2304.03442.pdf.
        :param half_life: When the elapsed time reaches this value, recency is 0.5,
        whereas recency is 1.0 when the elapsed time is zero.
        :param multiplicative: Whether the score should be calculated by weighted multiplication of the
        different components rather than weighted addition.
        :param decay_type: The type of decay applied to the recency estimation.
        :param alpha_recency: The weight of the recency score.
        :param alpha_importance: The weight of the importance score.
        :param alpha_relevance: The weight of the relevance score.
        """
        self.multiplicative = multiplicative
        self.alpha_relevance = alpha_relevance
        self.alpha_importance = alpha_importance
        self.alpha_recency = alpha_recency
        self.decay_type = decay_type
        self.decay_coefficient = self._decay_coefficient(decay_type, half_life)
        self.warned_importance_none = False
        self.warned_bad_elapsed = False

    @staticmethod
    def _decay_coefficient(decay_type: DecayType, half_life: float):
        if decay_type == DecayType.EXPONENTIAL:
            return -math.log(0.5) / half_life
        elif decay_type == DecayType.INVERSE:
            return half_life
        else:
            raise ValueError(f'Unrecognized decay type: {decay_type}')

    def _recency(self, elapsed: float):
        dt = self.decay_type
        if dt == DecayType.EXPONENTIAL:
            return math.exp(-(elapsed * self.decay_coefficient))
        elif dt == DecayType.INVERSE:
            dc = self.decay_coefficient
            return dc / (dc + elapsed)
        else:
            raise ValueError(f'Unrecognized decay type: {dt}')

    def _get_score(self, recency: float, importance: float, relevance: float, eps=1e-20):
        if self.multiplicative:
            if recency < 0:
                raise RuntimeError(f'Invalid recency: {recency}')
            if importance < 0:
                raise RuntimeError(f'Invalid importance: {importance}')
            if relevance < 0:
                raise RuntimeError(f'Invalid relevance: {relevance}')
            recency_value = math.log(recency + eps)
            importance_value = math.log(importance + eps)
            relevance_value = math.log(relevance + eps)
        else:
            recency_value = recency
            importance_value = importance
            relevance_value = relevance
        return recency_value * self.alpha_recency + \
            importance_value * self.alpha_importance + \
            relevance_value * self.alpha_relevance

    def rerank(self, r_memories: List[RetrievedMemory], mem: BaseTextMemory) -> List[RetrievedMemory]:
        if not mem.has_importance_model():
            raise RuntimeError('This reranker requires a memory with an importance model')
        scored_list = []
        current_time = time.time()
        for m in r_memories:
            importance = m.importance
            if importance is None:
                if not self.warned_importance_none:
                    logging.warning('Importance value is None.')
                    self.warned_importance_none = True
                importance = 0
            elapsed = current_time - m.timestamp
            if elapsed < 0:
                if not self.warned_bad_elapsed:
                    logging.warning(f'Time elapsed for memory is invalid: {elapsed:.4g} seconds.')
                    self.warned_bad_elapsed = True
                elapsed = 0
            recency = self._recency(elapsed)
            score = self._get_score(recency, importance, m.relevance)
            scored_list.append((score, m))
        scored_list.sort(key=lambda _t: _t[0], reverse=True)
        return [m for _, m in scored_list]

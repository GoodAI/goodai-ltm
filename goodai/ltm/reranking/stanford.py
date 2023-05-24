import enum
import logging
import math
import time
from typing import List, Optional
from goodai.ltm.mem.base import BaseReranker, RetrievedMemory, BaseTextMemory, BaseImportanceModel
from goodai.text_gen.base import BaseTextGenerationModel
from goodai.text_gen.openai_tg import OpenAICompletionModel


class DecayType(enum.Enum):
    EXPONENTIAL = 0
    INVERSE = 1


class StanfordReranker(BaseReranker):
    def __init__(self, half_life: float,
                 multiplicative: bool = False,
                 decay_type: DecayType = DecayType.EXPONENTIAL,
                 use_importance: bool = True,
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
        :param use_importance: Whether the memory's importance metric should be used.
        """
        self.use_importance = use_importance
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

    def _get_score(self, recency: float, importance: Optional[float], relevance: float, eps=1e-20):
        if self.multiplicative:
            if recency < 0:
                raise RuntimeError(f'Invalid recency: {recency}')
            if self.use_importance and (importance is None or importance < 0):
                raise RuntimeError(f'Invalid importance: {importance}')
            if relevance < 0:
                raise RuntimeError(f'Invalid relevance: {relevance}')
            recency_value = math.log(recency + eps)
            importance_value = math.log(importance + eps) if self.use_importance else None
            relevance_value = math.log(relevance + eps)
        else:
            recency_value = recency
            importance_value = importance
            relevance_value = relevance
        score = importance_value * self.alpha_importance if self.use_importance else 0
        return score + recency_value * self.alpha_recency + \
            relevance_value * self.alpha_relevance

    def rerank(self, r_memories: List[RetrievedMemory], mem: BaseTextMemory) -> List[RetrievedMemory]:
        if self.use_importance and not mem.has_importance_model():
            raise RuntimeError('This reranker requires a memory with an importance model')
        scored_list = []
        current_time = time.time()
        for m in r_memories:
            importance = m.importance
            if importance is None and self.use_importance:
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


class StanfordImportanceModel(BaseImportanceModel):
    def __init__(self, text_gen_model: BaseTextGenerationModel = None, prompt_template: str = None):
        if text_gen_model is None:
            text_gen_model = OpenAICompletionModel('text-davinci-003', max_tokens=2)
        if prompt_template is None:
            # Prompt from https://arxiv.org/pdf/2304.03442.pdf.
            prompt_template = "On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, " \
                     "making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), " \
                     "rate the likely poignancy of the following piece of memory.\r\n" \
                     "Memory: {mem_text}\r\nRating: "
        self.prompt_template = prompt_template
        self.text_gen_model = text_gen_model

    def get_importance(self, mem_text: str, min_value=1.0, max_value=10.0) -> float:
        template_params = {'mem_text': mem_text}
        prompt = self.prompt_template.format(**template_params)
        response = self.text_gen_model.generate(prompt)
        try:
            response_number = float(response.strip())
            response_number = max(min_value, min(max_value, response_number))
            return (response_number - min_value) / (max_value - min_value)
        except ValueError:
            logging.warning(f'Response from text generation model ("{response}") could not be converted to a number.')
            return 0

from typing import Tuple

import numpy as np
from goodai.ltm.data.query_passage.qa import QAQueryPassageDataSource

from goodai.ltm.data.query_passage.data_source import BaseQueryPassageDataSource
from transformers import PreTrainedTokenizer


class AutoQueryPassageDataSource:
    """
    Factory class for training and evaluation data sources for query-passage matching.
    """

    def __init__(self):
        pass

    @staticmethod
    def create(name: str, random: np.random.RandomState, tokenizer: PreTrainedTokenizer,
               max_query_tokens=40, min_passage_tokens=24,
               max_passage_tokens=36) -> Tuple[BaseQueryPassageDataSource, BaseQueryPassageDataSource]:
        """
        :param name: The dataset name
        :param random: An instance of a numpy RandomState object
        :param tokenizer: A Huggingface tokenizer
        :param max_query_tokens: The maximum number of query tokens
        :param min_passage_tokens: The minimum number of passage tokens
        :param max_passage_tokens: The maximum number of passage tokens
        :return: Tuple of train and test data sources
        """
        common_params = dict(
            random=random, tokenizer=tokenizer,
            max_query_tokens=max_query_tokens,
            min_passage_tokens=min_passage_tokens,
            max_passage_tokens=max_passage_tokens
        )
        if name in ['coqa', 'squad_v2', 'adversarial_qa']:
            return QAQueryPassageDataSource.create_data_sources(ds_name=name, **common_params)
        else:
            raise ValueError(f'Unknown: {name}')

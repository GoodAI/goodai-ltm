from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable

import numpy as np
from datasets import load_dataset

from goodai.helpers.tokenizer_helper import get_sentence_punctuation_ids
from goodai.ltm.data.query_passage.example import QueryPassageExample
from transformers import PreTrainedTokenizer

from goodai.ltm.data.names import NameSource
from goodai.ltm.data.query_passage.data_source import BaseQueryPassageDataSource


@dataclass
class SharcEntry:
    questionIds: List[int]
    snippetIds: List[int]
    scenarioIds: List[int]

    @staticmethod
    def from_example(example: dict, tokenizer: PreTrainedTokenizer) -> 'SharcEntry':
        question = example['question']
        scenario = example['scenario']
        snippet = example['snippet']
        snippet = snippet.strip()
        while len(snippet) > 0 and snippet[0] == '#':
            snippet = snippet[1:]
        question_ids = tokenizer.encode(' ' + question, add_special_tokens=False)
        scenario_ids = tokenizer.encode(scenario, add_special_tokens=False)
        snippet_ids = tokenizer.encode(snippet, add_special_tokens=False)
        return SharcEntry(question_ids, snippet_ids, scenario_ids)


class SharcQueryPassageDataSource(BaseQueryPassageDataSource):
    def __init__(self, random: np.random.RandomState, tokenizer: PreTrainedTokenizer,
                 qa_examples: List[dict],
                 max_query_tokens: int, min_passage_tokens: int, max_passage_tokens: int,
                 min_query_tokens=6):
        super().__init__()
        if max_query_tokens < min_query_tokens:
            raise ValueError(f'Expected max_query_tokens to be at least {min_query_tokens}')
        self.qa_examples = qa_examples
        self.min_query_tokens = min_query_tokens
        self.max_passage_tokens = max_passage_tokens
        self.min_passage_tokens = min_passage_tokens
        self.max_query_tokens = max_query_tokens
        self.tokenizer = tokenizer
        self.random = random
        self.tokenization_cache: Dict[int, SharcEntry] = dict()
        self.punctuation_ids = get_sentence_punctuation_ids(tokenizer,
                                                            include_line_break=False, include_colon=True)

    @classmethod
    def prepare_dataset(cls, ds_name: str):
        ds = load_dataset(ds_name)
        train_data = ds['train']
        test_data = ds.get('test')
        if test_data is None:
            test_data = ds['validation']
        train_data = cls._filter_valid(cls._process(train_data, ds_name))
        test_data = cls._filter_valid(cls._process(test_data, ds_name))
        return {
            'train': list(train_data),
            'test': list(test_data),
        }

    @classmethod
    def _add_id_column(cls, data):
        return data.add_column('id', [str(i) for i in range(data.num_rows)])

    @classmethod
    def _filter_valid(cls, entries: List[dict]):
        def _is_valid(entry: dict) -> bool:
            snippet = entry.get('snippet')
            return snippet is not None and len(snippet) > 0

        return [e for e in entries if _is_valid(e)]

    @classmethod
    def _process(cls, entries: Iterable[dict], ds_name: str) -> List[dict]:
        def _extend(e: dict):
            new_dict = dict(e)
            new_dict['ds_name'] = ds_name
            return new_dict

        return [_extend(entry) for entry in entries]

    @classmethod
    def create_data_sources(cls, random: np.random.RandomState, ds_name: str, tokenizer: PreTrainedTokenizer,
                            max_query_tokens: int, min_passage_tokens: int, max_passage_tokens: int) ->\
            Tuple['SharcQueryPassageDataSource', 'SharcQueryPassageDataSource']:
        data = cls.prepare_dataset(ds_name)
        train_examples = data['train']
        test_examples = data['test']
        train_ds = cls(random, tokenizer, train_examples, max_query_tokens, min_passage_tokens, max_passage_tokens)
        test_ds = cls(random, tokenizer, test_examples, max_query_tokens, min_passage_tokens, max_passage_tokens)
        return train_ds, test_ds,

    def get_tokenization(self, index: int) -> SharcEntry:
        tok_entry = self.tokenization_cache.get(index)
        if tok_entry is None:
            index = self.random.randint(0, len(self.qa_examples))
            qa_example: dict = self.qa_examples[index]
            tok_entry = SharcEntry.from_example(qa_example, self.tokenizer)
            self.tokenization_cache[index] = tok_entry
        return tok_entry

    def get_query_token_ids(self, tok_entry: SharcEntry, ends_with_name_p=0.75, use_names_p=0.95,
                            query_noise_p=0.75):
        r = self.random
        name2 = None
        if r.uniform() < use_names_p:
            name1, name2 = tuple(NameSource.get_instance().sample_first_names(r, 2))
        else:
            name1 = r.choice(['User', 'Player'])
        has_query_noise = self.random.uniform() < query_noise_p
        name_lead = ('\n' if has_query_noise else '') + name1 + ':'
        name1_ids = self.tokenizer.encode(name_lead)
        token_ids = name1_ids + tok_entry.questionIds
        ends_with_name = r.uniform() < ends_with_name_p
        if ends_with_name:
            if name2 is None:
                name2 = r.choice(['AI', 'Assistant', 'Agent', 'NPC'])
            name2_ids = self.tokenizer.encode('\n' + name2 + ':')
            token_ids += name2_ids
        if has_query_noise and len(tok_entry.scenarioIds) > 1:
            min_token_ids = min(self.max_query_tokens, max(self.min_query_tokens, len(token_ids)))
            num_tokens = r.randint(min_token_ids, self.max_query_tokens + 1)
            name_scenario_ids = name1_ids + tok_entry.scenarioIds
            full_sequence = name_scenario_ids + token_ids
            num_noise_tokens = max(0, num_tokens - len(token_ids))
            return full_sequence[-num_tokens:], num_noise_tokens,
        else:
            return token_ids[-self.max_query_tokens:], 0,

    def get_passage_token_ids(self, tok_entry: SharcEntry):
        r = self.random
        context_ids = tok_entry.snippetIds
        excess = len(context_ids) - self.max_passage_tokens
        if excess > 0:
            idx = r.randint(0, excess + 1)
            return context_ids[idx:idx + self.max_passage_tokens]
        else:
            return context_ids

    def sample_item(self, is_match: bool) -> QueryPassageExample:
        r = self.random
        n = len(self.qa_examples)
        for attempt in range(100):
            pos_index = r.randint(0, n)
            pos_tok_entry = self.get_tokenization(pos_index)
            query_token_ids, nqn = self.get_query_token_ids(pos_tok_entry)
            if is_match:
                passage_tok_entry = pos_tok_entry
            else:
                neg_index = r.randint(0, n)
                if neg_index == pos_index:
                    continue
                neg_tok_entry = self.get_tokenization(neg_index)
                passage_tok_entry = neg_tok_entry
            passage_token_ids = self.get_passage_token_ids(passage_tok_entry)
            return QueryPassageExample(query_token_ids, passage_token_ids, is_match)

        raise SystemError('Unable to find suitable qa_example!')

    def sample_items(self, count: int, approx_positive_fraction: float = 0.5) -> List[QueryPassageExample]:
        rnd_samples = self.random.uniform(size=count)
        is_match = rnd_samples <= approx_positive_fraction
        return [self.sample_item(is_match[i]) for i, _ in enumerate(range(count))]

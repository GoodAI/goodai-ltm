import logging
from typing import List, Dict, Tuple, Iterable

import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from goodai.ltm.data.query_passage.example import QueryPassageExample
from transformers import PreTrainedTokenizer

from goodai.ltm.data.names import NameSource
from goodai.ltm.data.query_passage.data_source import BaseQueryPassageDataSource
from goodai.ltm.data.query_passage.qa_tok_entry import QATokenizedEntry
from goodai.ltm.helpers.tokenizer_helper import get_sentence_punctuation_ids


class QAQueryPassageDataSource(BaseQueryPassageDataSource):
    def __init__(self, random: np.random.RandomState, tokenizer: PreTrainedTokenizer,
                 qa_examples: List[dict],
                 max_query_tokens: int, min_passage_tokens: int, max_passage_tokens: int,
                 min_query_tokens=6, min_answer_start_leeway: int = 16, answer_leeway_min_tokens: int = 32):
        super().__init__()
        self.answer_leeway_min_tokens = answer_leeway_min_tokens
        self.min_answer_start_leeway = min_answer_start_leeway
        if max_query_tokens < min_query_tokens:
            raise ValueError(f'Expected max_query_tokens to be at least {min_query_tokens}')
        self.qa_examples = qa_examples
        self.min_query_tokens = min_query_tokens
        self.max_passage_tokens = max_passage_tokens
        self.min_passage_tokens = min_passage_tokens
        self.max_query_tokens = max_query_tokens
        self.tokenizer = tokenizer
        self.random = random
        self.tokenization_cache: Dict[int, QATokenizedEntry] = dict()
        self.punctuation_ids = get_sentence_punctuation_ids(tokenizer,
                                                            include_line_break=False, include_colon=True)

    @classmethod
    def _prepare_coqa_eval(cls, data: Dataset) -> Dataset:
        data = cls._add_id_column(data)

        def process_example(example):
            # Currently, use only the first question / answer and discard follow-up dialog
            example['question'] = example['questions'][0]
            example['answers'] = {'text': [example['answers']['input_text'][0]],
                                  'answer_start': [example['answers']['answer_start'][0]]}
            return example

        return data.map(process_example, remove_columns=['questions'])

    @classmethod
    def prepare_dataset(cls, ds_name: str):
        if ds_name == 'adversarial_qa':
            ds = load_dataset(ds_name, 'adversarialQA')
        else:
            ds = load_dataset(ds_name)
        if ds_name == 'coqa':
            # Use only coca_fiction
            coqa_fiction = ds.filter(lambda example: example['source'] in {'gutenberg', 'mctest'})
            train_test = coqa_fiction['train'].train_test_split(train_size=.9)
            ds = DatasetDict({
                'train': cls._prepare_coqa_eval(train_test['train']),
                'test': cls._prepare_coqa_eval(train_test['test']),
            })
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
            answers = entry['answers']
            a_text = answers['text']
            return len(a_text) > 0

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
            Tuple['QAQueryPassageDataSource', 'QAQueryPassageDataSource']:
        data = cls.prepare_dataset(ds_name)
        train_examples = data['train']
        test_examples = data['test']
        train_ds = cls(random, tokenizer, train_examples, max_query_tokens, min_passage_tokens, max_passage_tokens)
        test_ds = cls(random, tokenizer, test_examples, max_query_tokens, min_passage_tokens, max_passage_tokens)
        return train_ds, test_ds,

    def get_tokenization(self, index: int) -> QATokenizedEntry:
        tok_entry = self.tokenization_cache.get(index)
        if tok_entry is None:
            index = self.random.randint(0, len(self.qa_examples))
            qa_example: dict = self.qa_examples[index]
            tok_entry = QATokenizedEntry.from_example(qa_example, self.tokenizer)
            self.tokenization_cache[index] = tok_entry
        return tok_entry

    def get_query_token_ids(self, tok_entry: QATokenizedEntry, ends_with_name_p=0.75, use_names_p=0.95,
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
        token_ids = name1_ids + tok_entry.question_token_ids
        ends_with_name = r.uniform() < ends_with_name_p
        if ends_with_name:
            if name2 is None:
                name2 = r.choice(['AI', 'Assistant', 'Agent', 'NPC'])
            name2_ids = self.tokenizer.encode('\n' + name2 + ':')
            token_ids += name2_ids
        if has_query_noise:
            min_token_ids = min(self.max_query_tokens, max(self.min_query_tokens, len(token_ids)))
            num_tokens = r.randint(min_token_ids, self.max_query_tokens + 1)
            full_sequence = tok_entry.content_token_ids + token_ids
            num_noise_tokens = max(0, num_tokens - len(token_ids))
            return full_sequence[-num_tokens:], num_noise_tokens,
        else:
            return token_ids[-self.max_query_tokens:], 0,

    def get_adjusted_answer_start(self, context_ids: List[int], answer_start: int, num_chunk_tokens: int):
        answer_start_leeway = self.min_answer_start_leeway + max(0, num_chunk_tokens - self.answer_leeway_min_tokens)
        found_index = max(0, answer_start - answer_start_leeway)
        for i in range(1, answer_start_leeway + 1):
            token_index = answer_start - i
            if token_index < 0:
                found_index = 0
                break
            token_id = context_ids[token_index]
            if token_id in self.punctuation_ids:
                found_index = token_index + 1
                break
        adjustment = answer_start - found_index
        extra_leeway = max(0, num_chunk_tokens - adjustment - self.answer_leeway_min_tokens)
        if extra_leeway > 0:
            offset = self.random.randint(0, extra_leeway + 1)
            found_index = max(0, found_index - offset)
        return found_index

    def get_passage_token_ids(self, tok_entry: QATokenizedEntry, is_match: bool, non_answer: bool,
                              non_answer_exclude_last_n: int):
        r = self.random
        context_ids = tok_entry.content_token_ids
        answer_start = tok_entry.answer_seq_index
        num_tokens = r.randint(self.min_passage_tokens, self.max_passage_tokens + 1)
        adj_answer_start = self.get_adjusted_answer_start(context_ids, answer_start, num_tokens)
        if is_match:
            passage_start = adj_answer_start
        elif non_answer:
            num_tokens_right = len(context_ids) - (answer_start + self.max_passage_tokens) - non_answer_exclude_last_n
            num_tokens_left = answer_start - self.max_passage_tokens
            if num_tokens_right >= num_tokens_left:
                p_from = answer_start + self.max_passage_tokens
                p_to = len(context_ids) - non_answer_exclude_last_n - num_tokens + 1
            else:
                p_from = 0
                p_to = answer_start - self.max_passage_tokens - num_tokens + 1
            if p_from >= p_to:
                return None
            passage_start = r.randint(p_from, p_to)
        else:
            p_to = len(context_ids) - num_tokens + 1
            if p_to < 1:
                return None
            passage_start = r.randint(0, p_to)
        passage_end = passage_start + num_tokens
        result = context_ids[passage_start:passage_end]
        if len(result) < self.min_passage_tokens:
            return None
        return result

    def sample_item(self, use_different_p=0.5) -> QueryPassageExample:
        r = self.random
        n = len(self.qa_examples)
        for attempt in range(100):
            pos_index = r.randint(0, n)
            pos_tok_entry = self.get_tokenization(pos_index)
            if pos_tok_entry.answer_seq_index == -1:
                logging.warning(f'Did not find location of answer excerpt in example with id {pos_tok_entry.e_id}')
                continue
            query_token_ids, nqn = self.get_query_token_ids(pos_tok_entry)
            is_match = r.choice([True, False])
            if is_match:
                passage_tok_entry = pos_tok_entry
                non_answer = False
            else:
                use_different = r.uniform() < use_different_p
                if use_different:
                    neg_index = r.randint(0, n)
                    neg_tok_entry = self.get_tokenization(neg_index)
                    non_answer = neg_index == pos_index
                    passage_tok_entry = neg_tok_entry
                else:
                    non_answer = True
                    passage_tok_entry = pos_tok_entry
            passage_token_ids = self.get_passage_token_ids(passage_tok_entry, is_match, non_answer, nqn)
            if passage_token_ids is None:
                continue
            return QueryPassageExample(query_token_ids, passage_token_ids, is_match)

        raise SystemError('Unable to find suitable qa_example!')

    def sample_items(self, count: int) -> List[QueryPassageExample]:
        return [self.sample_item() for _ in range(count)]

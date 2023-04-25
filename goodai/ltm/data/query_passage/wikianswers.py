import codecs
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from goodai.helpers.file_helper import download_zip
from goodai.helpers.tokenizer_helper import get_sentence_punctuation_ids
from goodai.ltm.data.query_passage.example import QueryPassageExample
from transformers import PreTrainedTokenizer
from goodai.ltm.data.names import NameSource
from goodai.ltm.data.query_passage.data_source import BaseQueryPassageDataSource


_data_url = 'https://github.com/GoodAI/goodai-ltm-artifacts/releases/download/data/summarized-wikianswers.zip'
_data_file_name = 'summarized-wikianswers.json'


@dataclass
class WikiAnswersEntry:
    questionIds: List[List[int]]
    answerIds: List[int]

    @staticmethod
    def from_example(example: dict, tokenizer: PreTrainedTokenizer) -> 'WikiAnswersEntry':
        questions = example['questions']
        answer = example['answer']
        question_ids = [tokenizer.encode(' ' + q, add_special_tokens=False) for q in questions]
        answer_ids = tokenizer.encode(' ' + answer, add_special_tokens=False)
        return WikiAnswersEntry(question_ids, answer_ids)


class WikiAnswersQueryPassageDataSource(BaseQueryPassageDataSource):
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
        self.tokenization_cache: Dict[int, WikiAnswersEntry] = dict()
        self.punctuation_ids = get_sentence_punctuation_ids(tokenizer,
                                                            include_line_break=False, include_colon=True)

    @classmethod
    def create_data_sources(cls, train_fraction: float, random: np.random.RandomState, tokenizer: PreTrainedTokenizer,
                            max_query_tokens: int, min_passage_tokens: int, max_passage_tokens: int) ->\
            Tuple['WikiAnswersQueryPassageDataSource', 'WikiAnswersQueryPassageDataSource']:
        data_dir = download_zip(_data_url)
        data_file = os.path.join(data_dir, _data_file_name)
        with codecs.open(data_file, 'r', 'utf-8') as fd:
            data = json.load(fd)
        random.shuffle(data)
        train_len = round(len(data) * train_fraction)
        train_entries = data[:train_len]
        test_entries = data[train_len:]
        train_ds = cls(random, tokenizer, train_entries, max_query_tokens, min_passage_tokens, max_passage_tokens)
        test_ds = cls(random, tokenizer, test_entries, max_query_tokens, min_passage_tokens, max_passage_tokens)
        return train_ds, test_ds,

    def get_tokenization(self, index: int) -> WikiAnswersEntry:
        tok_entry = self.tokenization_cache.get(index)
        if tok_entry is None:
            index = self.random.randint(0, len(self.qa_examples))
            qa_example: dict = self.qa_examples[index]
            tok_entry = WikiAnswersEntry.from_example(qa_example, self.tokenizer)
            self.tokenization_cache[index] = tok_entry
        return tok_entry

    def get_query_token_ids(self, tok_entry: WikiAnswersEntry, ends_with_name_p=0.75, use_names_p=0.95):
        r = self.random
        name2 = None
        if r.uniform() < use_names_p:
            name1, name2 = tuple(NameSource.get_instance().sample_first_names(r, 2))
        else:
            name1 = r.choice(['User', 'Player'])
        name_lead = name1 + ':'
        name1_ids = self.tokenizer.encode(name_lead, add_special_tokens=False)
        question_ids = tok_entry.questionIds
        nq = len(question_ids)
        if nq == 0:
            return None
        q_idx = r.randint(0, nq)
        selected_q_ids = question_ids[q_idx]
        token_ids = name1_ids + selected_q_ids
        ends_with_name = r.uniform() < ends_with_name_p
        if ends_with_name:
            if name2 is None:
                name2 = r.choice(['AI', 'Assistant', 'Agent', 'NPC'])
            name2_ids = self.tokenizer.encode('\n' + name2 + ':', add_special_tokens=False)
            token_ids += name2_ids
        return token_ids[-self.max_query_tokens:]

    def get_passage_token_ids(self, tok_entry: WikiAnswersEntry):
        r = self.random
        answer_ids = tok_entry.answerIds
        na = len(answer_ids)
        if na < self.min_passage_tokens:
            return None
        excess = na - self.max_passage_tokens
        if excess > 0:
            idx = r.randint(0, excess + 1)
            return answer_ids[idx:idx + self.max_passage_tokens]
        else:
            return answer_ids

    def sample_item(self, is_match: bool) -> QueryPassageExample:
        r = self.random
        n = len(self.qa_examples)
        for attempt in range(100):
            pos_index = r.randint(0, n)
            pos_tok_entry = self.get_tokenization(pos_index)
            query_token_ids = self.get_query_token_ids(pos_tok_entry)
            if query_token_ids is None:
                continue
            if is_match:
                passage_tok_entry = pos_tok_entry
            else:
                neg_index = r.randint(0, n)
                if neg_index == pos_index:
                    continue
                neg_tok_entry = self.get_tokenization(neg_index)
                passage_tok_entry = neg_tok_entry
            passage_token_ids = self.get_passage_token_ids(passage_tok_entry)
            if passage_token_ids is None:
                continue
            return QueryPassageExample(query_token_ids, passage_token_ids, is_match)
        raise SystemError('Unable to find suitable qa_example!')

    def sample_items(self, count: int, approx_positive_fraction: float = 0.5) -> List[QueryPassageExample]:
        rnd_samples = self.random.uniform(size=count)
        is_match = rnd_samples <= approx_positive_fraction
        return [self.sample_item(is_match[i]) for i, _ in enumerate(range(count))]

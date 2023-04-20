import re
from typing import List, Optional, Any, Dict, Tuple

import numpy as np
from transformers import PreTrainedTokenizer

from goodai.ltm.data.query_passage.data_source import BaseQueryPassageDataSource
from goodai.ltm.data.query_passage.example import QueryPassageExample


class WikiQueryPassageDataSource(BaseQueryPassageDataSource):
    def __init__(self, random: np.random.RandomState, tokenizer: PreTrainedTokenizer, articles: List[Any],
                 max_query_tokens: int, min_passage_tokens: int, max_passage_tokens: int,
                 gap_leeway=12, min_query_tokens=16):
        super().__init__()
        if max_query_tokens < min_query_tokens:
            raise ValueError(f'Expected max_query_tokens to be at least {min_query_tokens}')
        self.min_query_tokens = min_query_tokens
        self.max_passage_tokens = max_passage_tokens
        self.min_passage_tokens = min_passage_tokens
        self.max_query_tokens = max_query_tokens
        self.gap_leeway = gap_leeway
        self.min_par_len = max_query_tokens + max_passage_tokens + gap_leeway
        self.tokenizer = tokenizer
        self.articles = articles
        self.random = random
        self.pos_article_token_ids: Optional[List[List[int]]] = None
        self.neg_article_token_ids: Optional[List[List[int]]] = None
        self.current_par_idx = -1
        self.tokenization_cache: Dict[int, List[List[int]]] = dict()
        self.par_pattern = re.compile(r'\r?\n\r?\n')

    @classmethod
    def create_data_sources(cls, train_fraction: float, random: np.random.RandomState, tokenizer: PreTrainedTokenizer,
                            max_query_tokens: int, min_passage_tokens: int, max_passage_tokens: int,) ->\
            Tuple['WikiQueryPassageDataSource', 'WikiQueryPassageDataSource']:
        # TODO access to sampled wikipedia data
        wd = WikiData.get_instance()
        articles = list(wd.get_articles())
        random.shuffle(articles)
        train_len = round(len(articles) * train_fraction)
        train_articles = articles[:train_len]
        test_articles = articles[train_len:]
        train_ds = cls(random, tokenizer, train_articles, max_query_tokens, min_passage_tokens, max_passage_tokens)
        test_ds = cls(random, tokenizer, test_articles, max_query_tokens, min_passage_tokens, max_passage_tokens)
        return train_ds, test_ds,

    def get_tokenization(self, index: int) -> List[List[int]]:
        art_token_ids = self.tokenization_cache.get(index)
        if art_token_ids is None:
            index = self.random.randint(0, len(self.articles))
            article = self.articles[index]
            content = article['text']
            paragraphs: List[str] = self.par_pattern.split(content)
            art_token_ids = []
            for p in paragraphs:
                par_token_ids = self.tokenizer.encode(p, add_special_tokens=False)
                if len(par_token_ids) >= self.min_par_len:
                    art_token_ids.append(par_token_ids)
            self.tokenization_cache[index] = art_token_ids
        return art_token_ids

    def is_new_article_needed(self):
        return self.pos_article_token_ids is None or self.current_par_idx >= len(self.pos_article_token_ids)

    def init_current_article(self):
        pos_art_token_ids: List[List[int]] = []
        neg_art_token_ids: List[List[int]] = []
        for attempt in range(20):
            pos_index = self.random.randint(0, len(self.articles))
            pos_art_token_ids = self.get_tokenization(pos_index)
            if len(pos_art_token_ids) == 0:
                continue
            neg_index = self.random.randint(0, len(self.articles))
            if neg_index == pos_index:
                continue
            neg_art_token_ids = self.get_tokenization(neg_index)
            if len(neg_art_token_ids) == 0:
                continue
            break
        if len(pos_art_token_ids) == 0 or len(neg_art_token_ids) == 0:
            raise SystemError('Unable to find article with suitable paragraph lengths.')
        self.neg_article_token_ids = neg_art_token_ids
        self.pos_article_token_ids = pos_art_token_ids
        self.current_par_idx = 0

    def _get_neg_passage_ids(self, num_passage_tokens: int, p_use_different_art: float = 0.5):
        r = self.random
        use_different = r.uniform() < p_use_different_art
        if not use_different:
            from_par_idx = self.current_par_idx + 2
            to_par_idx = len(self.pos_article_token_ids)
            if from_par_idx < to_par_idx:
                neg_par_idx = r.randint(from_par_idx, to_par_idx)
                par_token_ids = self.pos_article_token_ids[neg_par_idx]
                flex = len(par_token_ids) - num_passage_tokens
                if flex >= 0:
                    start = r.randint(0, flex + 1)
                    return par_token_ids[start:start + num_passage_tokens]
        neg_par_id = r.randint(0, len(self.neg_article_token_ids))
        neg_par_token_ids = self.neg_article_token_ids[neg_par_id]
        flex = len(neg_par_token_ids) - num_passage_tokens
        passage_start = r.randint(0, flex + 1)
        passage_end = passage_start + num_passage_tokens
        passage_ids = neg_par_token_ids[passage_start:passage_end]
        return passage_ids

    def sample_items(self, count: int, approx_positive_fraction: float = 0.5) -> List[QueryPassageExample]:
        r = self.random
        result = []
        for i in range(count):
            if self.is_new_article_needed():
                self.init_current_article()
            par_token_ids = self.pos_article_token_ids[self.current_par_idx]
            num_query_tokens = r.randint(self.min_query_tokens, self.max_query_tokens + 1)
            num_gap_tokens = r.randint(0, self.gap_leeway + 1)
            num_passage_tokens = r.randint(self.min_passage_tokens, self.max_passage_tokens + 1)
            is_match = self.random.uniform() <= approx_positive_fraction
            if is_match:
                total_tokens = num_query_tokens + num_gap_tokens + num_passage_tokens
            else:
                total_tokens = num_query_tokens
            flex = len(par_token_ids) - total_tokens
            start = r.randint(0, flex + 1)
            query_end = start + num_query_tokens
            query_ids = par_token_ids[start:query_end]
            if is_match:
                passage_start = query_end + num_gap_tokens
                passage_end = passage_start + num_passage_tokens
                passage_ids = par_token_ids[passage_start:passage_end]
            else:
                passage_ids = self._get_neg_passage_ids(num_passage_tokens)
            result.append(QueryPassageExample(query_ids, passage_ids, is_match))
            self.current_par_idx += 1
        return result

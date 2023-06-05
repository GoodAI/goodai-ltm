from enum import Enum
from typing import Optional, List, Any

from transformers import PreTrainedTokenizer


class ChunkExpansionLimitType(Enum):
    SENTENCE = 0,
    LINE = 1,
    PARAGRAPH = 2,
    SECTION = 3

    @staticmethod
    def batch_encode(tokenizer: PreTrainedTokenizer, texts: List[str]):
        tok = tokenizer.batch_encode_plus(texts, add_special_tokens=False, return_attention_mask=False)
        return tok['input_ids']

    @staticmethod
    def distinct(a_list: List[List[Any]]) -> List[List[Any]]:
        tuples = [tuple(x) for x in a_list]
        tuples = list(set(tuples))
        return [list(x) for x in tuples]

    def get_token_ids(self, tokenizer: PreTrainedTokenizer) -> List[List[int]]:
        if self == self.SENTENCE:
            return self.batch_encode(tokenizer, ['.', '?', '!', '???', '!!!'])
        elif self == self.LINE:
            cr_id = tokenizer.encode('\r', add_special_tokens=False)
            nl_id = tokenizer.encode('\n', add_special_tokens=False)
            result = self.batch_encode(tokenizer, ['\n', '\r\n']) + \
                [cr_id + nl_id]
            return self.distinct(result)
        elif self == self.PARAGRAPH:
            cr_id = tokenizer.encode('\r', add_special_tokens=False)
            nl_id = tokenizer.encode('\n', add_special_tokens=False)
            result = self.batch_encode(tokenizer, ['\n\n', '\r\n\r\n']) + \
                [nl_id + nl_id] + [cr_id + nl_id + cr_id + nl_id]
            return self.distinct(result)
        elif self == self.SECTION:
            return []
        else:
            raise ValueError(f'Unhandled limit type: {self}')


class ChunkExpansionConfig:
    def __init__(self, min_extra_side_tokens: int = 0, max_extra_side_tokens: int = 24,
                 limit_type: ChunkExpansionLimitType = ChunkExpansionLimitType.SENTENCE):
        self.limit_type = limit_type
        self.min_extra_side_tokens = min_extra_side_tokens
        self.max_extra_side_tokens = max_extra_side_tokens

    @classmethod
    def for_sentence(cls, max_extra_side_tokens: int = 24):
        return cls(max_extra_side_tokens=max_extra_side_tokens, limit_type=ChunkExpansionLimitType.SENTENCE)

    @classmethod
    def for_line_break(cls, min_extra_side_tokens: int = 16, max_extra_side_tokens: int = 64):
        return cls(min_extra_side_tokens=min_extra_side_tokens,
                   max_extra_side_tokens=max_extra_side_tokens,
                   limit_type=ChunkExpansionLimitType.LINE)

    @classmethod
    def for_paragraph(cls, max_extra_side_tokens: int = 192):
        return cls(max_extra_side_tokens=max_extra_side_tokens, limit_type=ChunkExpansionLimitType.PARAGRAPH)

    @classmethod
    def for_section(cls, max_extra_side_tokens: int = 1024):
        return cls(max_extra_side_tokens=max_extra_side_tokens, limit_type=ChunkExpansionLimitType.SECTION)

    @classmethod
    def for_chunk(cls):
        return cls(max_extra_side_tokens=0, limit_type=ChunkExpansionLimitType.SECTION)


class TextMemoryConfig:
    chunk_capacity: int
    """
    The capacity of a chunk, in tokens.
    """

    queue_capacity: int
    """
    The capacity of the memory's chunk queue, i.e. the maximum number of chunks it can hold.
    """

    reranking_k_factor: float
    """
    When a reranking mechanism is available, the value of k passed to the retrieval function is
    multiplied by reranking_k_factor and the resulting number of chunks is retrieved from
    the chunk store before applying the reranking procedure.
    """

    max_query_length: Optional[int]
    """
    The maximum length of queries used to produce embeddings. The left side of queries
    exceeding this maximum length are truncated (after rewriting). 
    """

    chunk_overlap_fraction: float
    """
    The fraction of a chunk's length that overlaps with the previous and next chunk.
    This can be a value from 0 to 0.5.
    """

    redundancy_overlap_threshold: float
    """
    The fraction of a retrieved passage's length that causes it to be considered 
    redundant if it overlaps with other passages that are a better match to a query.
    """

    chunk_expansion_config: ChunkExpansionConfig
    """
    The chunk expansion configuration.
    """

    def __init__(self):
        self.max_query_length = 40  # Tokens
        self.chunk_capacity = 24  # Tokens
        self.queue_capacity = 64000  # Chunks
        self.reranking_k_factor = 10.0
        self.chunk_overlap_fraction = 0.5  # 0 to 0.5
        self.redundancy_overlap_threshold = 0.75
        self.chunk_expansion_config = ChunkExpansionConfig(max_extra_side_tokens=self.chunk_capacity)

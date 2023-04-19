from dataclasses import dataclass
from typing import List, Optional, Tuple
from transformers import PreTrainedTokenizer

from goodai.ltm.helpers.tokenizer_helper import get_token_index


@dataclass
class QATokenizedEntry:
    ds_name: str
    e_id: str
    context: str
    answer_token_ids: List[int]
    answer_seq_index: int
    question: str
    content_token_ids: Optional[List[int]] = None
    question_token_ids: Optional[List[int]] = None

    @staticmethod
    def from_example(example: dict, tokenizer: PreTrainedTokenizer) -> 'QATokenizedEntry':
        ds_name = example['ds_name']
        e_id = example['id']
        context = example.get('context')
        if context is None:
            context = example.get('story')
        question = example.get('question')
        question_ids = tokenizer.encode(' ' + question, add_special_tokens=False)
        answers = example['answers']
        answer_text_list = answers['text']
        answer_text = answer_text_list[0]
        answer_index = answers['answer_start'][0]
        tok_output = tokenizer(context, return_offsets_mapping=True, add_special_tokens=False)
        context_ids = tok_output['input_ids']
        offset_mapping: List[Tuple[int, int]] = tok_output['offset_mapping']
        answer_token_index = get_token_index(context_ids, offset_mapping, answer_index)
        answer_token_ids = tokenizer.encode(' ' + answer_text, add_special_tokens=False)
        return QATokenizedEntry(ds_name=ds_name, e_id=e_id, context=context,
                                content_token_ids=context_ids,
                                answer_token_ids=answer_token_ids,
                                answer_seq_index=answer_token_index,
                                question_token_ids=question_ids,
                                question=question)

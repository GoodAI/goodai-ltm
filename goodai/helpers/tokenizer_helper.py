import torch
import torch.nn.utils.rnn as R
from typing import Set, List, Tuple, Optional
from transformers import PreTrainedTokenizer


def get_pad_token_id(tokenizer: PreTrainedTokenizer):
    p_tid = tokenizer.pad_token_id
    if p_tid is None:
        tokens = tokenizer.tokenize(' ')
        if len(tokens) == 0:
            tokens = tokenizer.tokenize('-')
            if len(tokens) == 0:
                raise Exception('Unable to find suitable padding token')
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        p_tid = token_ids[0]
    return p_tid


def get_eos_token_id(tokenizer: PreTrainedTokenizer):
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = get_pad_token_id(tokenizer)
    return eos_id


def get_sentence_punctuation_ids(tokenizer: PreTrainedTokenizer, include_line_break=False,
                                 include_colon=False) -> Set[int]:
    def _update(text: str, r_set: set):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == 1:
            r_set.add(token_ids[0])

    result = set()
    _update('.', result)
    _update('!', result)
    _update('!!', result)
    _update('!!!', result)
    _update('?', result)
    _update('??', result)
    _update('???', result)
    if include_line_break:
        _update('\n', result)
    if include_colon:
        _update(':', result)
    return result


def get_token_index(token_ids: List[int], token_offsets: List[Tuple[int, int]], offset: int):
    for i, (token_id, offset_pair) in enumerate(zip(token_ids, token_offsets)):
        o_from, o_to = offset_pair
        if o_from <= offset <= o_to:
            return i
    return -1


def get_model_inputs(input_ids_list: List[List[int]], pad_id: int, device: torch.device,
                     min_seq_len: Optional[int] = None, prefix: str = '',
                     tokenizer: PreTrainedTokenizer = None, return_token_lengths: bool = False):
    # Note: padding on the right
    max_seq_len = max(len(seq) for seq in input_ids_list)
    if min_seq_len is None:
        min_seq_len = max_seq_len
    else:
        min_seq_len = max(max_seq_len, min_seq_len)
    input_ids_list = [ids + [pad_id] * (min_seq_len - len(ids)) for ids in input_ids_list]
    input_ids = torch.as_tensor(input_ids_list, dtype=torch.int64, device=device)
    attention_mask = (input_ids != pad_id).float()
    result = {
        prefix + 'input_ids': input_ids,
        prefix + 'attention_mask': attention_mask,
    }
    if return_token_lengths:
        def _lengths(seq: List[int]):
            tokens = tokenizer.convert_ids_to_tokens(seq)
            return [len(t) for t in tokens]

        if tokenizer is None:
            raise ValueError('tokenizer cannot be None when token lengths are requested')
        token_lengths = [_lengths(seq) for seq in input_ids_list]
        result[prefix + 'token_lengths'] = torch.as_tensor(token_lengths, dtype=torch.long, device=device)
    return result


def get_attention_after_token(input_ids: torch.Tensor, attention_mask: torch.Tensor, token_lengths: torch.Tensor,
                              token_id: int, exclude_last_n_chars: int, device: torch.device) -> torch.Tensor:
    att_lengths = token_lengths * attention_mask
    inc_lengths = torch.flip(torch.cumsum(torch.flip(att_lengths, dims=(1,)), dim=1), dims=(1,))
    considered: torch.Tensor = torch.greater(inc_lengths, exclude_last_n_chars)
    matches_token: torch.Tensor = torch.eq(input_ids, token_id) & considered
    index_range = torch.arange(0, input_ids.size(1), device=device).unsqueeze(0)
    indices = torch.where(matches_token, index_range, -1)
    max_indices, _ = torch.max(indices, dim=1, keepdim=True)
    at_att_mask = torch.greater(index_range, max_indices).float()
    return at_att_mask


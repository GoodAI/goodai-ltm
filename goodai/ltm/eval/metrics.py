from typing import List

from transformers import PreTrainedTokenizer


def levenshtein_distance(s_pred: List[str], t_exp: List[str]):
    m = len(s_pred)
    n = len(t_exp)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s_pred[i - 1] == t_exp[j - 1]:
                sub_cost = 0
            else:
                sub_cost = 1
            d[i][j] = min([
                d[i - 1][j] + 1,  # Deletion
                d[i][j - 1] + 1,  # Insertion
                d[i - 1][j - 1] + sub_cost,  # Substitution
            ])
    return d[m][n]


def get_correctness_score(tokenizer: PreTrainedTokenizer, predicted_answer: str, expected_answer: str):
    predicted_raw_tokens = tokenizer.tokenize(' ' + predicted_answer, add_special_tokens=False)
    return _get_correctness_score_for_tokens_ea(tokenizer, predicted_raw_tokens, expected_answer)


def _get_correctness_score_for_tokens_ea(tokenizer: PreTrainedTokenizer, predicted_raw_tokens: List[str],
                                         expected_answer: str):
    expected_raw_tokens_2 = tokenizer.tokenize(' ' + expected_answer, add_special_tokens=False)
    correctness_2 = _get_correctness_score_for_tokens(predicted_raw_tokens, expected_raw_tokens_2)
    return correctness_2


def _get_correctness_score_for_tokens(predicted_raw_tokens: List[str], expected_raw_tokens: List[str]):
    predicted_tokens = _norm_tokens(predicted_raw_tokens)
    expected_tokens = _norm_tokens(expected_raw_tokens)
    if len(expected_tokens) == 0:
        return 0.0
    else:
        s_distance = _subseq_distance(predicted_tokens, expected_tokens)
        max_distance = len(expected_tokens)
        return 100.0 * (1 - s_distance / max_distance)


def _subseq_distance(s_pred: List[str], t_exp: List[str]):
    min_s_len = len(t_exp)
    i_max = len(s_pred) - min_s_len
    if i_max <= 0:
        pred_subseqs = [s_pred]
    else:
        pred_subseqs = [s_pred[i:i + min_s_len] for i in range(i_max + 1)]
    distances = [levenshtein_distance(s, t_exp) for s in pred_subseqs]
    return min(distances)


def _norm_token(token: str):
    if token.startswith('Ġ'):
        token = token[1:]
    token = token.lower()
    if token in [',', '!', ';', '..', '...', '!!', '!!!']:
        token = '.'
    return token


def _norm_tokens(raw_tokens: List[str]):
    tokens = [_norm_token(t) for t in raw_tokens]
    tokens = [t for t in tokens if t != '']
    if tokens[-1:] == ['.']:
        tokens = tokens[:-1]
    return tokens

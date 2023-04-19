import unittest

from transformers import PreTrainedTokenizer, AutoTokenizer

from goodai.ltm.eval.metrics import get_correctness_score


class TestMetrics(unittest.TestCase):
    tokenizer: PreTrainedTokenizer

    def setUp(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')

    def test_exact_match_01(self):
        exp_t = "The sun is at the center of the solar system."
        pred_t = "the sun is at the center of the solar system"
        c = get_correctness_score(self.tokenizer, pred_t, exp_t)
        assert c >= 99.0

    def test_match_01(self):
        exp_t = "The sun is at the center of the solar system."
        pred_t = "the sun is a star at the center of the solar system"
        c = get_correctness_score(self.tokenizer, pred_t, exp_t)
        assert c >= 60

    def test_match_02(self):
        exp_t = "The sun is at the center of the solar system."
        pred_t = "Our star, the sun, is at the center of the solar system and the Earth is about 8 light minutes away!"
        c = get_correctness_score(self.tokenizer, pred_t, exp_t)
        assert c >= 70

    def test_mismatch_01(self):
        exp_t = "The sun is at the center of the solar system."
        pred_t = "It was the best of times, it was the worst of times!"
        c = get_correctness_score(self.tokenizer, pred_t, exp_t)
        assert c <= 30

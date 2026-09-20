import pytest

from benchmarks.metrics import similarity_metrics
from benchmarks.providers import Inference, ScoreResult
from benchmarks.stsb_jev import RUBRIC, evaluate_one, identity


class FakeProvider:
    def infer(self, state, question):
        return Inference(
            ScoreResult(4.5, {str(index): probability for index, probability in enumerate((0, 0, 0, 0, 0.5, 0.5))}, {str(index): text for index, text in enumerate(RUBRIC)}),
            {"provider": "fake"},
        )


def test_evaluate_one_formats_the_pair_and_preserves_float_label():
    record = evaluate_one(
        FakeProvider(),
        3,
        {"sentence1": "A person is running.", "sentence2": "Someone runs.", "label": 4.8},
    )
    assert record["state"] == "Sentence 1: A person is running.\nSentence 2: Someone runs."
    assert record["expected"] == 4.8
    assert record["result"]["score"] == 4.5
    assert len(record["question"]["criteria"]) == 6


def test_identity_and_metrics_match_stsb():
    assert identity()["dataset"]["config"] == "stsb"
    records = [
        {"expected": 0.0, "result": {"score": 1.0}},
        {"expected": 2.0, "result": {"score": 3.0}},
        {"expected": 5.0, "result": {"score": 4.0}},
    ]
    result = similarity_metrics(records)
    assert result["pearson"] == pytest.approx(0.9538209665)
    assert result["spearman"] == 1.0

from types import SimpleNamespace

import pytest
from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer

from benchmarks.providers import Choice, Noul, Score, TransformersNoulProvider, TypeSafeProvider


class FakeClient:
    def __init__(self, answer):
        self.answer = answer

    def system_one(self, **kwargs):
        return SimpleNamespace(
            answers={"answer": self.answer},
            model="jev-1.13.0",
            usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 3, "output_tokens": 1}),
        )


def provider_with(answer):
    provider = object.__new__(TypeSafeProvider)
    provider.model = "jev-1.13.0"
    provider.client = FakeClient(answer)
    return provider


def test_typesafe_choice_conversion():
    answer = ChoiceAnswer(type="choice", choice="a", confidence=0.7, probabilities={"a": 0.7, "b": 0.3})
    inference = provider_with(answer).infer("state", Choice("pick", ("a", "b")))
    assert inference.result.as_dict() == {"type": "choice", "choice": "a", "probabilities": {"a": 0.7, "b": 0.3}}
    assert inference.metadata["confidence"] == 0.7


def test_typesafe_rejects_invalid_distribution():
    answer = ChoiceAnswer(type="choice", choice="a", confidence=0.7, probabilities={"a": 0.7, "b": 0.4})
    with pytest.raises(ValueError, match="sum to one"):
        provider_with(answer).infer("state", Choice("pick", ("a", "b")))


def test_typesafe_normalizes_provider_rounding():
    answer = ChoiceAnswer(type="choice", choice="a", confidence=0.7, probabilities={"a": 0.7, "b": 0.29})
    probabilities = provider_with(answer).infer("state", Choice("pick", ("a", "b"))).result.probabilities
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert probabilities == pytest.approx({"a": 0.7 / 0.99, "b": 0.29 / 0.99})


def test_typesafe_score_and_noul_conversion():
    score = ScoreAnswer(
        type="score", score=1.7, confidence=0.8,
        probabilities={0: 0.1, 1: 0.1, 2: 0.8}, legend={0: "low", 1: "medium", 2: "high"},
    )
    result = provider_with(score).infer("state", Score("rate", ("low", "medium", "high"))).result.as_dict()
    assert result == {
        "type": "score", "score": 1.7,
        "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
        "legend": {"0": "low", "1": "medium", "2": "high"},
    }

    noul = NoulAnswer(type="noul", noul=0.9)
    assert provider_with(noul).infer("state", Noul("yes?")).result.as_dict() == {"type": "noul", "noul": 0.9}


def test_typesafe_rejects_score_outside_rubric():
    answer = ScoreAnswer(
        type="score", score=3.1, confidence=0.8,
        probabilities={0: 0.0, 1: 0.0, 2: 1.0}, legend={0: "low", 1: "medium", 2: "high"},
    )
    with pytest.raises(ValueError, match="rubric"):
        provider_with(answer).infer("state", Score("rate", ("low", "medium", "high")))


class FakeLogits:
    def __getitem__(self, key):
        assert key == 0
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [-1.0, 1.0]


class FakeTransformersModel:
    def __call__(self, **kwargs):
        assert kwargs == {"input_ids": [1, 2, 3]}
        return SimpleNamespace(logits=FakeLogits())


def test_transformers_noul_provider_returns_positive_probability():
    provider = object.__new__(TransformersNoulProvider)
    provider.model_id = "example/model"
    provider.revision = "abc123"
    provider.positive_index = 1
    provider.tokenizer = lambda text, **kwargs: {"input_ids": [1, 2, 3]}
    provider.model = FakeTransformersModel()

    inference = provider.infer("A wonderful film.", Noul("positive?"))

    assert inference.result.noul == pytest.approx(0.8807970779)
    assert inference.metadata == {
        "provider": "huggingface",
        "model": "example/model",
        "revision": "abc123",
        "confidence": pytest.approx(0.8807970779),
    }

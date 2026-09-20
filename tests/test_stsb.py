from pathlib import Path

import pytest

import benchmarks.stsb as stsb
from benchmarks.providers import Inference, ScalarScoreResult, ScoreResult


ROWS = [
    {"sentence1": "A person runs.", "sentence2": "Someone is running.", "score": 4.5},
    {"sentence1": "A cat sleeps.", "sentence2": "Markets rose.", "score": 0.2},
]


class FakeSentenceTransformer:
    def infer_pair(self, sentence1, sentence2):
        assert (sentence1, sentence2) == ("A person runs.", "Someone is running.")
        return Inference(ScalarScoreResult(0.75), {"provider": "fake"})


class FakeTypeSafe:
    def infer(self, state, question):
        return Inference(
            ScoreResult(4.2, {str(i): 1 / 6 for i in range(6)}, {str(i): text for i, text in enumerate(stsb.RUBRIC)}),
            {"provider": "fake"},
        )


def test_sentence_transformer_record_preserves_pair_and_scalar_result():
    record = stsb.evaluate_sentence_transformer(FakeSentenceTransformer(), 7, ROWS[0])
    assert record["dataset_id"] == 7
    assert record["state"] == "Sentence 1: A person runs.\nSentence 2: Someone is running."
    assert record["expected"] == 4.5
    assert record["result"] == {"type": "score", "score": 0.75}
    assert "probabilities" not in record["result"]


def test_typesafe_uses_the_same_row_and_frozen_rubric():
    record = stsb.evaluate_typesafe(FakeTypeSafe(), 7, ROWS[0])
    assert record["expected"] == 4.5
    assert record["question"]["criteria"] == list(stsb.RUBRIC)


def test_identity_pins_original_test_rows_model_and_separate_paths(tmp_path: Path, monkeypatch):
    local = stsb.identity("sentence-transformers")
    jev = stsb.identity("typesafe")
    assert local["benchmark"] == "stsb"
    assert local["dataset"] == jev["dataset"]
    assert local["dataset"]["split"] == "test"
    assert local["dataset"]["rows"] == 1_379
    assert len(local["dataset"]["row_digest"]) == 64
    assert local["model_revision"] == stsb.MODEL_REVISION
    assert local["model_card_status"] == "deprecated"
    monkeypatch.setattr(stsb, "ROOT", tmp_path)
    local_args = stsb.parse_args(["--provider", "sentence-transformers"])
    jev_args = stsb.parse_args(["--provider", "typesafe"])
    assert local_args.output != jev_args.output


def test_row_digest_is_order_and_label_sensitive():
    digest = stsb.row_digest(ROWS)
    assert digest != stsb.row_digest(list(reversed(ROWS)))
    changed = [*ROWS[:1], {**ROWS[1], "score": 0.3}]
    assert digest != stsb.row_digest(changed)


def test_dataset_validation_checks_count_and_digest(monkeypatch):
    monkeypatch.setattr(stsb, "EXPECTED_ROWS", len(ROWS))
    monkeypatch.setattr(stsb, "ROW_DIGEST", stsb.row_digest(ROWS))
    stsb.validate_dataset(ROWS)
    with pytest.raises(ValueError, match="row digest mismatch"):
        stsb.validate_dataset(list(reversed(ROWS)))

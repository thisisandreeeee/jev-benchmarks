import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import banking77
from benchmarks.providers import ChoiceResult, Inference, Space2Provider, _space2_choice

LABELS = ("one", "two")
ROWS = [{"text": "first", "label": 0}, {"text": "second", "label": 1}]


class FakeProvider:
    def infer(self, state, question):
        probabilities = {"one": 0.8, "two": 0.2} if state == "first" else {"one": 0.1, "two": 0.9}
        return Inference(ChoiceResult(max(probabilities, key=probabilities.get), probabilities), {"provider": "fake"})


def author_data(rows=ROWS):
    return {
        f"banking-{index}": {
            "turns": [
                {
                    "text": row["text"],
                    "label": {"DEFAULT_DOMAIN": {LABELS[row["label"]]: {}}},
                    "extra_info": {"intent_label": row["label"]},
                }
            ]
        }
        for index, row in enumerate(rows)
    }


def test_space2_choice_converts_logits_and_rejects_bad_output():
    result = _space2_choice([-1.0, 1.0], LABELS)
    assert result.choice == "two"
    assert result.probabilities == pytest.approx({"one": 0.119202922, "two": 0.880797078})
    with pytest.raises(ValueError, match="label mapping"):
        _space2_choice([1.0], LABELS)


def test_space2_preprocessing_keeps_the_last_fifty_wordpieces():
    provider = object.__new__(Space2Provider)
    provider.tokenizer = type(
        "Tokenizer",
        (),
        {
            "tokenize": staticmethod(lambda token: [token]),
            "convert_tokens_to_ids": staticmethod(lambda tokens: list(range(len(tokens)))),
        },
    )()
    assert provider._tokens(" ".join(f"word{index}" for index in range(60))) == [13, *range(10, 60), 7]


def test_manifest_contains_complete_pinned_release():
    manifest = banking77.load_manifest(banking77.DEFAULT_MANIFEST)
    assert manifest["repository"]["commit"] == "188835d4f9948563a6b9c8ac50cd0f3ae4021ed6"
    assert len(manifest["label_mapping"]) == len(set(manifest["label_mapping"])) == 77
    assert manifest["checkpoint"]["sha256"] == manifest["files"]["checkpoint"]["sha256"]
    assert len(manifest["manifest_sha256"]) == 64


def test_release_validation_rejects_digest_mismatch(tmp_path: Path):
    artifact = tmp_path / "checkpoint"
    artifact.write_bytes(b"checkpoint")
    manifest = {
        "files": {
            "checkpoint": {
                "path": "checkpoint",
                "size": len(b"checkpoint"),
                "sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            }
        }
    }
    assert banking77.validate_release(tmp_path, manifest)["checkpoint"] == artifact
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        banking77.validate_release(tmp_path, manifest)


def test_alignment_uses_stable_author_ids_and_complete_mapping(tmp_path: Path):
    path = tmp_path / "test.json"
    path.write_text(json.dumps(author_data()))
    banking77.validate_alignment(ROWS, LABELS, path, LABELS)

    changed = author_data()
    changed["banking-1"]["turns"][0]["text"] = "changed"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="canonical row 1"):
        banking77.validate_alignment(ROWS, LABELS, path, LABELS)


def test_row_digest_is_order_and_label_sensitive():
    digest = banking77.row_digest(ROWS, LABELS)
    assert digest != banking77.row_digest(list(reversed(ROWS)), LABELS)
    assert digest != banking77.row_digest(ROWS, tuple(reversed(LABELS)))


def test_both_providers_share_dataset_identity_and_separate_paths(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(banking77, "ROOT", tmp_path)
    jev = banking77.identity("typesafe", LABELS)
    manifest = {
        "repository": {},
        "paper": "paper",
        "checkpoint": {},
        "preprocessing": {},
        "label_mapping": list(LABELS),
        "manifest_sha256": "digest",
    }
    space2 = banking77.identity("space-2", LABELS, manifest)
    assert jev["dataset"] == space2["dataset"]
    assert banking77.parse_args(["--provider", "typesafe"]).output != banking77.parse_args(
        ["--provider", "space-2"]
    ).output


def test_complete_space2_run_publishes_to_provider_path(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(banking77, "ROOT", tmp_path)
    monkeypatch.setattr(banking77, "EXPECTED_ROWS", len(ROWS))
    monkeypatch.setattr(banking77, "ROW_DIGEST", banking77.row_digest(ROWS, LABELS))
    manifest = {
        "repository": {},
        "paper": "paper",
        "checkpoint": {},
        "preprocessing": {},
        "label_mapping": list(LABELS),
        "manifest_sha256": "digest",
    }
    result = banking77.run(
        ROWS,
        LABELS,
        FakeProvider(),
        "space-2",
        tmp_path / "run",
        manifest=manifest,
        limit=None,
        resume=False,
        concurrency=1,
    )
    published = tmp_path / "results" / "banking77" / "space-2" / "state_epoch_51.json"
    assert result["status"] == "complete"
    assert json.loads(published.read_text())["metrics"]["accuracy"] == 1.0


def test_space2_rejects_parallel_runner():
    with pytest.raises(SystemExit):
        banking77.parse_args(["--provider", "space-2", "--concurrency", "2"])

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks import banking77, clinc150, hwu64, space2_intents
from benchmarks.providers import ChoiceResult, Inference, Space2Provider, _space2_choice

LABELS = ("one", "two")
ROWS = [{"text": "first", "label": "one"}, {"text": "second", "label": "two"}]
CONFIG = replace(
    banking77.CONFIG,
    benchmark="fake",
    expected_rows=2,
    expected_labels=2,
    row_digest=space2_intents.row_digest(ROWS),
    space2_model="checkpoint",
    author_prefix="fake-",
)


class FakeProvider:
    def infer(self, state, question):
        probabilities = {"one": 0.8, "two": 0.2} if state == "first" else {"one": 0.1, "two": 0.9}
        return Inference(ChoiceResult(max(probabilities, key=probabilities.get), probabilities), {"provider": "fake"})


def author_data(rows=ROWS, prefix="fake-"):
    return {
        f"{prefix}{index}": {
            "turns": [
                {
                    "text": row["text"],
                    "label": {"DEFAULT_DOMAIN": {row["label"]: {}}},
                    "extra_info": {"intent_label": LABELS.index(row["label"])},
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


@pytest.mark.parametrize(
    ("config", "count", "model"),
    [(banking77.CONFIG, 77, "state_epoch_51"), (clinc150.CONFIG, 150, "state_epoch_27"), (hwu64.CONFIG, 64, "state_epoch_25")],
)
def test_manifests_contain_complete_pinned_releases(config, count, model):
    manifest = space2_intents.load_manifest(config.manifest, count)
    assert manifest["repository"]["commit"] == "188835d4f9948563a6b9c8ac50cd0f3ae4021ed6"
    assert len(manifest["label_mapping"]) == len(set(manifest["label_mapping"])) == count
    assert manifest["checkpoint"]["name"].endswith(f"{model}.model")
    assert manifest["checkpoint"]["sha256"] == manifest["files"]["checkpoint"]["sha256"]


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
    assert space2_intents.validate_release(tmp_path, manifest)["checkpoint"] == artifact
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        space2_intents.validate_release(tmp_path, manifest)


def test_alignment_uses_stable_ids_and_documented_casefold(tmp_path: Path):
    path = tmp_path / "test.json"
    path.write_text(json.dumps(author_data()))
    space2_intents.validate_alignment(ROWS, LABELS, path, LABELS, CONFIG)

    changed = author_data()
    changed["fake-1"]["turns"][0]["text"] = "SECOND"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="canonical row 1"):
        space2_intents.validate_alignment(ROWS, LABELS, path, LABELS, CONFIG)
    space2_intents.validate_alignment(ROWS, LABELS, path, LABELS, replace(CONFIG, casefold_author_text=True))


def test_row_digest_is_order_and_label_sensitive():
    digest = space2_intents.row_digest(ROWS)
    assert digest != space2_intents.row_digest(list(reversed(ROWS)))
    assert digest != space2_intents.row_digest([{**ROWS[0], "label": "two"}, ROWS[1]])


def test_both_providers_share_dataset_identity_and_separate_paths():
    manifest = {
        "repository": {},
        "paper": "paper",
        "checkpoint": {},
        "preprocessing": {},
        "label_mapping": list(LABELS),
        "manifest_sha256": "digest",
    }
    jev = space2_intents.identity("typesafe", LABELS, CONFIG)
    supervised = space2_intents.identity("space-2", LABELS, CONFIG, manifest)
    assert jev["dataset"] == supervised["dataset"]
    assert space2_intents.parse_args(CONFIG, ["--provider", "typesafe"]).output != space2_intents.parse_args(
        CONFIG, ["--provider", "space-2"]
    ).output


def test_complete_space2_run_publishes_to_provider_path(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(space2_intents, "ROOT", tmp_path)
    manifest = {
        "repository": {},
        "paper": "paper",
        "checkpoint": {},
        "preprocessing": {},
        "label_mapping": list(LABELS),
        "manifest_sha256": "digest",
    }
    result = space2_intents.run(
        ROWS,
        LABELS,
        FakeProvider(),
        "space-2",
        tmp_path / "run",
        CONFIG,
        manifest=manifest,
        limit=None,
        resume=False,
        concurrency=1,
    )
    published = tmp_path / "results" / "fake" / "space-2" / "checkpoint.json"
    assert result["status"] == "complete"
    assert json.loads(published.read_text())["metrics"]["accuracy"] == 1.0


def test_partial_run_resumes_without_duplicate_predictions(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(space2_intents, "ROOT", tmp_path)
    output = tmp_path / "run"
    first = space2_intents.run(
        ROWS,
        LABELS,
        FakeProvider(),
        "typesafe",
        output,
        CONFIG,
        manifest=None,
        limit=1,
        resume=False,
        concurrency=1,
    )
    second = space2_intents.run(
        ROWS,
        LABELS,
        FakeProvider(),
        "typesafe",
        output,
        CONFIG,
        manifest=None,
        limit=None,
        resume=True,
        concurrency=1,
    )
    records = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert first["status"] == "partial"
    assert second["status"] == "complete"
    assert sorted(record["dataset_id"] for record in records) == [0, 1]


def test_dataset_adapters_freeze_label_spaces_and_filters():
    class Dataset(list):
        pass

    clinc = Dataset([{"text": "a", "intent": 0}, {"text": "outside", "intent": 1}, {"text": "b", "intent": 2}])
    clinc.features = {"intent": type("Feature", (), {"names": ("alpha", "oos", "beta")})()}
    rows, labels = clinc150.load_rows(clinc, ("beta", "alpha"))
    assert rows == [{"text": "a", "label": "alpha"}, {"text": "b", "label": "beta"}]
    assert labels == ("beta", "alpha")

    rows, labels = hwu64.load_rows([{"utterance": "hello", "label": 1}], ("alpha", "beta"))
    assert rows == [{"text": "hello", "label": "beta"}]
    assert labels == ("alpha", "beta")


def test_space2_rejects_parallel_runner():
    with pytest.raises(SystemExit):
        space2_intents.parse_args(CONFIG, ["--provider", "space-2", "--concurrency", "2"])

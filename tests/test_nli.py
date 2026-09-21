import json
from pathlib import Path

import pytest

from benchmarks import banking77, sst2, space2_intents
from benchmarks.providers import (
    Choice,
    NliZeroShotProvider,
    Noul,
    _nli_choice,
    _nli_noul,
)
from benchmarks.providers import nli_manifest as nli

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = nli.load_manifest()
LABELS = ("a", "b")
VERBALIZATION = {"a": "alpha", "b": "beta"}


def test_nli_choice_softmaxes_entailment_across_candidates():
    result = _nli_choice([0.0, 2.0], LABELS)
    assert result.choice == "b"
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
    assert result.probabilities == pytest.approx({"a": 0.119202922, "b": 0.880797078})


def test_nli_choice_rejects_mismatched_or_nonfinite_scores():
    with pytest.raises(ValueError, match="candidate labels"):
        _nli_choice([1.0], LABELS)
    with pytest.raises(ValueError, match="finite"):
        _nli_choice([float("nan"), 1.0], LABELS)


def test_nli_noul_returns_positive_probability():
    assert _nli_noul([2.0, 0.0]).noul == pytest.approx(0.880797078)
    with pytest.raises(ValueError, match="two finite"):
        _nli_noul([1.0])


def test_entailment_index_is_read_from_the_checkpoint_config():
    assert NliZeroShotProvider._entailment_index({0: "entailment", 1: "not_entailment"}) == 0
    assert NliZeroShotProvider._entailment_index({"0": "contradiction", "1": "neutral", "2": "entailment"}) == 2
    with pytest.raises(ValueError, match="exactly one entailment"):
        NliZeroShotProvider._entailment_index({0: "a", 1: "b"})


def make_provider(labels, verbalization, table, positive_label=None):
    provider = object.__new__(NliZeroShotProvider)
    provider.model_id = "example/model"
    provider.revision = "rev123"
    provider.labels = tuple(labels)
    provider.verbalization = dict(verbalization)
    provider.template = "This text is about {}"
    provider.positive_label = positive_label
    provider.max_length = 512
    provider.batch_size = 4
    provider.device = "cpu"
    provider.prepared = {}
    provider._entailment_logits = lambda pairs: [table[hypothesis] for _, hypothesis in pairs]
    return provider


def test_choice_inference_maps_hypothesis_scores_to_labels():
    provider = make_provider(
        LABELS,
        VERBALIZATION,
        {"This text is about alpha": 0.0, "This text is about beta": 2.0},
    )
    inference = provider.infer("hello", Choice("pick", LABELS))
    assert inference.result.choice == "b"
    assert inference.metadata["provider"] == "nli-zero-shot"
    assert inference.metadata["confidence"] == pytest.approx(0.880797078)


def test_choice_inference_rejects_an_incomplete_candidate_set():
    provider = make_provider(LABELS, VERBALIZATION, {})
    with pytest.raises(ValueError, match="complete candidate label set"):
        provider.infer("hello", Choice("pick", ("a",)))


def test_choice_inference_caches_repeated_states():
    calls = []
    provider = make_provider(
        LABELS,
        VERBALIZATION,
        {"This text is about alpha": 0.0, "This text is about beta": 2.0},
    )
    provider._entailment_logits = lambda pairs: calls.append(len(pairs)) or [
        0.0 if "alpha" in hypothesis else 2.0 for _, hypothesis in pairs
    ]
    first = provider.infer("first", Choice("pick", LABELS))
    second = provider.infer("first", Choice("pick", LABELS))
    assert first.result == second.result
    assert calls == [2]


def test_noul_inference_returns_positive_probability():
    provider = make_provider(
        ("positive", "negative"),
        {"positive": "positive sentiment", "negative": "negative sentiment"},
        {"This text is about positive sentiment": 2.0, "This text is about negative sentiment": 0.0},
        positive_label="positive",
    )
    inference = provider.infer("A wonderful film.", Noul("positive?"))
    assert inference.result.noul == pytest.approx(0.880797078)
    assert inference.metadata["confidence"] == pytest.approx(0.880797078)


def test_verbalization_rule_is_deterministic():
    assert nli.verbalize_label("card_arrival") == "card arrival"
    assert nli.verbalize_label("Refund_not_showing_up") == "refund not showing up"
    assert nli.verbalize_label("reverted_card_payment?") == "reverted card payment"


def test_manifest_labels_match_the_pinned_benchmarks():
    for name in ("banking77", "clinc150", "hwu64"):
        span2 = json.loads((ROOT / "manifests" / f"space2-{name}.json").read_text())["label_mapping"]
        entry = MANIFEST["benchmarks"][name]
        assert set(entry["labels"]) == set(span2)
        assert {label: nli.verbalize_label(label) for label in entry["labels"]} == entry["verbalization"]
    assert MANIFEST["benchmarks"]["sst2"]["labels"] == ["positive", "negative"]


def test_manifest_rejects_a_verbalization_digest_mismatch(tmp_path: Path):
    value = json.loads((ROOT / "manifests" / "nli-labels.json").read_text())
    value["benchmarks"]["sst2"]["verbalization"]["positive"] = "good"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="digest mismatch"):
        nli.load_manifest(path)


def test_validate_labels_rejects_a_mismatched_label_set():
    with pytest.raises(ValueError, match="canonical label set"):
        nli.validate_labels({"labels": ["a"]}, ("a", "b"))


def test_nli_identity_shares_the_dataset_identity_with_jev():
    labels = tuple(MANIFEST["benchmarks"]["banking77"]["labels"])
    jev = space2_intents.identity("typesafe", labels, banking77.CONFIG)
    local = space2_intents.identity(
        "nli", labels, banking77.CONFIG, None, nli.provider_identity(MANIFEST, "banking77")
    )
    assert jev["dataset"] == local["dataset"]
    assert local["provider"] == "nli-zero-shot-local"
    assert local["model_revision"] == "cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2"
    assert local["hypothesis_template"] == "This text is about {}"
    assert local["candidates"] == jev["candidates"]


def test_nli_default_output_paths_are_separate():
    intent = space2_intents.parse_args(banking77.CONFIG, ["--provider", "nli"])
    assert intent.output == space2_intents.ROOT / "runs" / "banking77" / f"nli-{nli.SLUG}"
    sentiment = sst2.parse_args(["--provider", "nli"])
    assert sentiment.output == sst2.ROOT / "runs" / "sst2" / f"nli-{nli.SLUG}"


def test_nli_run_publishes_under_the_zero_shot_regime(tmp_path: Path, monkeypatch):
    from dataclasses import replace

    rows = [{"text": "first", "label": "a"}, {"text": "second", "label": "b"}]
    config = replace(
        banking77.CONFIG,
        benchmark="fake",
        expected_rows=2,
        expected_labels=2,
        row_digest=space2_intents.row_digest(rows),
        space2_model="checkpoint",
        author_prefix="fake-",
    )
    manifest = {
        "provider": "nli-zero-shot-local",
        "model": "example/model",
        "revision": "rev",
        "model_card": "card",
        "license": "mit",
        "hypothesis_template": "This text is about {}",
        "entailment_label": "entailment",
        "normalization": "softmax_over_candidates_of_entailment_logits",
        "max_length": 512,
        "verbalization_sha256": {"fake": "digest"},
        "benchmarks": {"fake": {"labels": ["a", "b"], "verbalization": {"a": "alpha", "b": "beta"}}},
    }
    captured = {}

    def fake_run_benchmark(dataset, provider, output, **kwargs):
        captured.update(kwargs)
        return {"status": "partial", "evaluated": 0, "total": len(dataset), "metrics": {}}

    monkeypatch.setattr(space2_intents, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(space2_intents, "ROOT", tmp_path)
    space2_intents.run(
        rows,
        ("a", "b"),
        make_provider(("a", "b"), {"a": "alpha", "b": "beta"}, {}),
        "nli",
        tmp_path / "run",
        config,
        manifest=None,
        limit=None,
        resume=False,
        concurrency=1,
        nli_manifest=manifest,
    )
    assert captured["result_path"] == (
        tmp_path / "results" / "fake" / "nli-deberta-v3-large-zeroshot-v2.0.json"
    )
    assert captured["identity"]["dataset"] == space2_intents.dataset_identity(config)


def test_sst2_nli_identity_pins_the_checkpoint():
    value = sst2.identity("nli")
    assert value["provider"] == "nli-zero-shot-local"
    assert value["model"] == MANIFEST["model"]
    assert value["model_revision"] == MANIFEST["revision"]
    assert value["max_length"] == 512

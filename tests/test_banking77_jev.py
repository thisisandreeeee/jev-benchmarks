import json
from pathlib import Path

import pytest

import benchmarks.banking77_jev as banking77
from benchmarks.banking77_jev import INSTRUCTION, atomic_json, metrics, publish, run_benchmark
from benchmarks.providers import ChoiceResult, Inference


LABELS = ("one", "two")
DATASET = [
    {"text": "first", "label": 0},
    {"text": "second", "label": 1},
    {"text": "third", "label": 0},
]


class FakeProvider:
    def __init__(self):
        self.calls = []

    def infer(self, state, question):
        self.calls.append(state)
        probabilities = {"one": 0.8, "two": 0.2} if state != "second" else {"one": 0.1, "two": 0.9}
        return Inference(ChoiceResult(max(probabilities, key=probabilities.get), probabilities), {"provider": "fake", "usage": {}})


class FailingProvider(FakeProvider):
    def infer(self, state, question):
        if state == "second":
            raise RuntimeError("nope")
        return super().infer(state, question)


def test_metrics():
    records = [
        {"expected": "one", "result": {"choice": "one", "probabilities": {"one": 0.8, "two": 0.2}}},
        {"expected": "two", "result": {"choice": "one", "probabilities": {"one": 0.6, "two": 0.4}}},
    ]
    result = metrics(records)
    assert result["accuracy"] == 0.5
    assert result["mean_confidence"] == pytest.approx(0.7)


def test_limited_run_resumes_without_duplicates(tmp_path: Path):
    provider = FakeProvider()
    run = run_benchmark(DATASET, LABELS, provider, tmp_path / "run", limit=2, resume=False, concurrency=1)
    assert run["status"] == "partial"
    assert provider.calls == ["first", "second"]

    run_benchmark(DATASET, LABELS, provider, tmp_path / "run", limit=2, resume=True, concurrency=1)
    assert provider.calls == ["first", "second"]
    records = [json.loads(line) for line in (tmp_path / "run" / "predictions.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["question"]["instructions"] == INSTRUCTION


def test_complete_run_publishes(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(banking77, "ROOT", tmp_path)
    run = run_benchmark(DATASET, LABELS, FakeProvider(), tmp_path / "run", limit=None, resume=False, concurrency=1)
    result = tmp_path / "results" / "banking77" / "typesafe" / "jev-1.13.0.json"
    assert run["status"] == "complete"
    assert json.loads(result.read_text())["evaluated"] == 3


def test_failure_keeps_successful_records(tmp_path: Path):
    with pytest.raises(RuntimeError, match="nope"):
        run_benchmark(DATASET, LABELS, FailingProvider(), tmp_path / "run", limit=None, resume=False, concurrency=1)
    run = json.loads((tmp_path / "run" / "run.json").read_text())
    assert run["status"] == "partial"
    assert run["evaluated"] == 2


def test_atomic_json_and_publication_conflict(tmp_path: Path):
    path = tmp_path / "run.json"
    atomic_json(path, {"ok": True})
    assert json.loads(path.read_text()) == {"ok": True}

    run = {
        "identity": {}, "dependencies": {}, "repository_revision": None,
        "evaluated": 1, "total": 1, "metrics": {}, "provider_statistics": {},
    }
    result = tmp_path / "result.json"
    publish(run, result)
    publish(run, result)
    result.write_text("{}")
    with pytest.raises(FileExistsError):
        publish(run, result)

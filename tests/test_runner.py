import json
from pathlib import Path

import pytest

from benchmarks.runner import read_predictions, run_benchmark


def record(row_id: int, value: str) -> dict:
    return {
        "dataset_id": row_id,
        "state": value,
        "expected": value,
        "result": {"type": "fake"},
        "provider_metadata": {"provider": "fake"},
    }


def metrics(records: list[dict]) -> dict[str, float]:
    return {"count": float(len(records))}


def test_resume_reconciles_run_after_final_prediction_was_written(tmp_path: Path):
    output = tmp_path / "run"
    result_path = tmp_path / "results" / "fake.json"
    dataset = ["first", "second"]

    def evaluate(_provider, row_id, row):
        return record(row_id, row)

    run_benchmark(
        dataset,
        object(),
        output,
        identity={"benchmark": "fake"},
        evaluate=evaluate,
        metrics=metrics,
        result_path=result_path,
        limit=None,
        resume=False,
        concurrency=1,
        dependencies={"test": "1"},
    )

    stale = json.loads((output / "run.json").read_text())
    stale["status"] = "partial"
    stale["evaluated"] = 1
    stale["metrics"] = {"count": 1.0}
    (output / "run.json").write_text(json.dumps(stale))

    def should_not_run(*_args):
        raise AssertionError("resume should not reevaluate complete predictions")

    resumed = run_benchmark(
        dataset,
        object(),
        output,
        identity={"benchmark": "fake"},
        evaluate=should_not_run,
        metrics=metrics,
        result_path=result_path,
        limit=None,
        resume=True,
        concurrency=1,
        dependencies={"test": "1"},
    )

    assert resumed["status"] == "complete"
    assert resumed["evaluated"] == 2
    assert resumed["metrics"] == {"count": 2.0}
    assert json.loads(result_path.read_text())["evaluated"] == 2


def test_read_predictions_rejects_duplicate_ids(tmp_path: Path):
    path = tmp_path / "predictions.jsonl"
    value = json.dumps(record(0, "first"))
    path.write_text(f"{value}\n{value}\n")

    with pytest.raises(ValueError, match="duplicate dataset IDs"):
        read_predictions(path)

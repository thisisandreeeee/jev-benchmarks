import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import compare


def choice(row_id: int, state: str, expected: str, predicted: str) -> dict:
    return {
        "dataset_id": row_id,
        "state": state,
        "expected": expected,
        "result": {"type": "choice", "choice": predicted, "probabilities": {}},
        "provider_metadata": {"provider": "fake"},
    }


def noul(row_id: int, state: str, expected: int, probability: float) -> dict:
    return {
        "dataset_id": row_id,
        "state": state,
        "expected": expected,
        "result": {"type": "noul", "noul": probability},
        "provider_metadata": {"provider": "fake"},
    }


def score(row_id: int, state: str, expected: float, predicted: float) -> dict:
    return {
        "dataset_id": row_id,
        "state": state,
        "expected": expected,
        "result": {"type": "score", "score": predicted},
        "provider_metadata": {"provider": "fake"},
    }


def make_run(
    tmp_path: Path,
    name: str,
    records: list[dict],
    *,
    benchmark: str = "fake",
    dataset: dict | None = None,
    metrics: dict | None = None,
    status: str = "complete",
    evaluated: int | None = None,
    total: int | None = None,
) -> Path:
    directory = tmp_path / name
    directory.mkdir(parents=True)
    total = len(records) if total is None else total
    dataset = (
        dataset
        if dataset is not None
        else {"id": "fake", "revision": "rev", "split": "test", "rows": total, "row_digest": "digest"}
    )
    run = {
        "identity": {"benchmark": benchmark, "dataset": dataset},
        "status": status,
        "evaluated": len(records) if evaluated is None else evaluated,
        "total": total,
        "metrics": metrics or {},
    }
    (directory / "run.json").write_text(json.dumps(run))
    (directory / "predictions.jsonl").write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    )
    return directory


def accuracy_records(counts: str) -> list[dict]:
    """Build rows from a two-character code per row: both, left, right, or wrong."""
    records = []
    for row_id, code in enumerate(counts):
        hit = code in ("b", "l")
        records.append(choice(row_id, f"row {row_id}", "yes", "yes" if hit else "no"))
    return records


def test_paired_accuracy_outcomes_and_difference(tmp_path: Path):
    left = make_run(
        tmp_path,
        "left",
        accuracy_records("bllwwb"),
        metrics={"accuracy": 4 / 6},
    )
    right_records = []
    for row_id, code in enumerate("bllwwb"):
        hit = code in ("b", "r")
        right_records.append(choice(row_id, f"row {row_id}", "yes", "yes" if hit else "no"))
    right = make_run(tmp_path, "right", right_records, metrics={"accuracy": 2 / 6})

    artifact = compare.compare(left, right, samples=500, seed=7)

    assert artifact["metric"] == "accuracy"
    assert artifact["rows"] == 6
    assert artifact["left"]["score"] == pytest.approx(4 / 6)
    assert artifact["right"]["score"] == pytest.approx(2 / 6)
    assert artifact["difference"]["value"] == pytest.approx(1 / 3)
    assert artifact["outcomes"] == {
        "both_correct": 2,
        "left_only": 2,
        "right_only": 0,
        "both_wrong": 2,
    }
    assert artifact["bootstrap"]["samples"] == 500
    assert artifact["bootstrap"]["seed"] == 7
    assert artifact["bootstrap"]["excluded_samples"] == 0


def test_identical_inputs_and_seed_are_byte_identical(tmp_path: Path):
    left = make_run(tmp_path, "left", accuracy_records("blbl"), metrics={"accuracy": 1.0})
    right_records = [
        choice(row_id, f"row {row_id}", "yes", "yes" if code in ("b", "r") else "no")
        for row_id, code in enumerate("blbl")
    ]
    right = make_run(tmp_path, "right", right_records, metrics={"accuracy": 2 / 4})

    first = compare.compare(left, right, samples=300, seed=3)
    second = compare.compare(left, right, samples=300, seed=3)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_bootstrap_is_deterministic_for_a_fixed_seed(tmp_path: Path):
    left_records = compare.load_run(make_run(tmp_path, "left", accuracy_records("blblbwbw")))["records"]
    right_records = compare.load_run(make_run(tmp_path, "right", accuracy_records("bwbwbwbw")))["records"]
    first = compare.bootstrap_accuracy(
        left_records, right_records, "choice", samples=200, seed=1, confidence=0.95
    )
    second = compare.bootstrap_accuracy(
        left_records, right_records, "choice", samples=200, seed=1, confidence=0.95
    )
    assert first == second


def test_tied_rank_spearman_comparison(tmp_path: Path):
    expected = [1.0, 1.0, 2.0, 3.0]
    left_positives = [4.0, 4.0, 2.0, 1.0]
    right_positives = [1.0, 1.0, 2.0, 4.0]
    left = make_run(
        tmp_path,
        "left",
        [score(i, f"row {i}", e, p) for i, (e, p) in enumerate(zip(expected, left_positives))],
        metrics={"spearman": -1.0},
    )
    right = make_run(
        tmp_path,
        "right",
        [score(i, f"row {i}", e, p) for i, (e, p) in enumerate(zip(expected, right_positives))],
        metrics={"spearman": 1.0},
    )

    artifact = compare.compare(left, right, samples=400, seed=11)

    assert artifact["metric"] == "spearman"
    assert artifact["outcomes"] is None
    assert artifact["left"]["score"] == pytest.approx(-1.0)
    assert artifact["right"]["score"] == pytest.approx(1.0)
    assert artifact["difference"]["value"] == pytest.approx(-2.0)
    assert artifact["difference"]["interval"]["lower"] <= -2.0
    assert artifact["difference"]["interval"]["upper"] == pytest.approx(-2.0)


def test_bootstrap_spearman_fails_when_expected_is_constant(tmp_path: Path):
    records = [score(i, f"row {i}", 1.0, float(i)) for i in range(4)]
    left = make_run(tmp_path, "left", records, metrics={"spearman": None})
    right = make_run(tmp_path, "right", records, metrics={"spearman": None})

    with pytest.raises(ValueError, match="too few valid bootstrap samples"):
        compare.bootstrap_spearman(
            compare.load_run(left)["records"],
            compare.load_run(right)["records"],
            samples=100,
            seed=1,
            confidence=0.95,
        )


def test_noul_accuracy_uses_probability_threshold(tmp_path: Path):
    left_records = [
        noul(0, "a", 1, 0.9),
        noul(1, "b", 1, 0.9),
        noul(2, "c", 1, 0.1),
    ]
    right_records = [
        noul(0, "a", 1, 0.9),
        noul(1, "b", 1, 0.1),
        noul(2, "c", 1, 0.9),
    ]
    left = make_run(tmp_path, "left", left_records, metrics={"accuracy": 2 / 3})
    right = make_run(tmp_path, "right", right_records, metrics={"accuracy": 2 / 3})

    artifact = compare.compare(left, right, samples=100, seed=1)

    assert artifact["metric"] == "accuracy"
    assert artifact["outcomes"] == {
        "both_correct": 1,
        "left_only": 1,
        "right_only": 1,
        "both_wrong": 0,
    }


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda run: run.update(status="partial"), "not a complete run"),
        (lambda run: run.update(evaluated=1, total=2), "incomplete"),
        (lambda run: run["identity"].update(benchmark="other"), "benchmark identities differ"),
        (
            lambda run: run["identity"]["dataset"].update(revision="other"),
            "dataset identities differ",
        ),
    ],
)
def test_rejects_incompatible_runs(tmp_path: Path, mutate, message):
    left = make_run(tmp_path, "left", accuracy_records("bl"), metrics={"accuracy": 1 / 2})
    right = make_run(tmp_path, "right", accuracy_records("bl"), metrics={"accuracy": 1 / 2})
    run = json.loads((right / "run.json").read_text())
    mutate(run)
    (right / "run.json").write_text(json.dumps(run))

    with pytest.raises(ValueError, match=message):
        compare.compare(left, right, samples=10, seed=1)


def test_rejects_different_row_counts(tmp_path: Path):
    identity = {"id": "fake", "revision": "rev", "split": "test"}
    left = make_run(tmp_path, "left", accuracy_records("bl"), dataset=identity, metrics={"accuracy": 1 / 2})
    right = make_run(tmp_path, "right", accuracy_records("blb"), dataset=identity, metrics={"accuracy": 2 / 3})
    with pytest.raises(ValueError, match="row counts differ"):
        compare.compare(left, right, samples=10, seed=1)


def test_rejects_mismatched_rows_and_expected_values(tmp_path: Path):
    left = make_run(tmp_path, "left", [choice(0, "a", "yes", "yes")], metrics={"accuracy": 1.0})
    right = make_run(tmp_path, "right", [choice(0, "b", "yes", "yes")], metrics={"accuracy": 1.0})
    with pytest.raises(ValueError, match="states differ"):
        compare.compare(left, right, samples=10, seed=1)

    right = make_run(tmp_path, "right-2", [choice(0, "a", "no", "yes")], metrics={"accuracy": 1.0})
    with pytest.raises(ValueError, match="expected values differ"):
        compare.compare(left, right, samples=10, seed=1)


def test_rejects_missing_and_duplicate_rows(tmp_path: Path):
    left = make_run(
        tmp_path,
        "left",
        [choice(0, "a", "yes", "yes")],
        total=2,
        evaluated=2,
        metrics={"accuracy": 1.0},
    )
    right = make_run(tmp_path, "right", accuracy_records("bb"), metrics={"accuracy": 1.0})
    with pytest.raises(ValueError, match="missing canonical rows"):
        compare.load_run(left)

    duplicate = json.dumps(choice(0, "a", "yes", "yes"))
    directory = tmp_path / "duplicate"
    directory.mkdir()
    (directory / "run.json").write_text(
        json.dumps(
            {
                "identity": {"benchmark": "fake", "dataset": {"rows": 2}},
                "status": "complete",
                "evaluated": 2,
                "total": 2,
                "metrics": {"accuracy": 1.0},
            }
        )
    )
    (directory / "predictions.jsonl").write_text(f"{duplicate}\n{duplicate}\n")
    with pytest.raises(ValueError, match="duplicate dataset IDs"):
        compare.load_run(directory)


def test_rejects_incompatible_metrics(tmp_path: Path):
    left = make_run(tmp_path, "left", [choice(0, "a", "yes", "yes")], metrics={"accuracy": 1.0})
    right = make_run(tmp_path, "right", [score(0, "a", 1.0, 1.0)], metrics={"spearman": None})
    with pytest.raises(ValueError, match="incompatible metrics"):
        compare.compare(left, right, samples=10, seed=1)


def test_rejects_identity_row_count_mismatch(tmp_path: Path):
    dataset = {"id": "fake", "rows": 9, "revision": "rev", "split": "test", "row_digest": "digest"}
    left = make_run(tmp_path, "left", [choice(0, "a", "yes", "yes")], dataset=dataset, metrics={"accuracy": 1.0})
    right = make_run(tmp_path, "right", [choice(0, "a", "yes", "yes")], dataset=dataset, metrics={"accuracy": 1.0})
    with pytest.raises(ValueError, match="identity row count"):
        compare.compare(left, right, samples=10, seed=1)


def test_rejects_stored_metric_mismatch(tmp_path: Path):
    left = make_run(tmp_path, "left", accuracy_records("bb"), metrics={"accuracy": 0.5})
    right = make_run(tmp_path, "right", accuracy_records("bb"), metrics={"accuracy": 1.0})
    with pytest.raises(ValueError, match="does not match recomputed"):
        compare.compare(left, right, samples=10, seed=1)


def test_publication_is_atomic_noop_and_refuses_conflicts(tmp_path: Path):
    artifact = {"schema_version": 1, "difference": {"value": 1.0}}
    path = tmp_path / "nested" / "comparison.json"

    compare.publish(artifact, path)
    written = path.read_bytes()
    assert json.loads(written) == artifact
    assert not list(path.parent.glob("*.tmp"))

    compare.publish(artifact, path)
    assert path.read_bytes() == written

    with pytest.raises(FileExistsError, match="refusing to overwrite conflicting comparison"):
        compare.publish({"schema_version": 2}, path)


def test_default_output_uses_benchmark_and_run_names():
    path = compare.default_output("sst2", Path("runs/sst2/typesafe-jev-1.13.0"), Path("runs/sst2/huggingface-roberta"))
    assert path == compare.ROOT / "results" / "sst2" / "comparisons" / "typesafe-jev-1.13.0--huggingface-roberta.json"


def test_parse_args_defaults_and_validation():
    args = compare.parse_args(["--left", "a", "--right", "b"])
    assert args.samples == compare.DEFAULT_SAMPLES
    assert args.seed == compare.DEFAULT_SEED
    assert args.confidence == compare.DEFAULT_CONFIDENCE
    assert args.output is None

    with pytest.raises(SystemExit):
        compare.parse_args(["--left", "a", "--right", "b", "--samples", "0"])
    with pytest.raises(SystemExit):
        compare.parse_args(["--left", "a", "--right", "b", "--confidence", "1"])


def test_main_publishes_explicit_output(tmp_path: Path):
    left = make_run(tmp_path, "left", accuracy_records("bl"), metrics={"accuracy": 1.0})
    right = make_run(tmp_path, "right", accuracy_records("bl"), metrics={"accuracy": 1.0})
    output = tmp_path / "comparison.json"

    assert compare.main(["--left", str(left), "--right", str(right), "--output", str(output), "--samples", "50"]) == 0
    assert json.loads(output.read_text())["rows"] == 2

    assert compare.main(["--left", str(left), "--right", str(right), "--output", str(output), "--samples", "50"]) == 0


def test_cli_module_does_not_import_a_model_sdk_or_provider():
    forbidden = {
        "torch",
        "transformers",
        "datasets",
        "sentence_transformers",
        "typesafe_sdk",
        "benchmarks.providers",
        "benchmarks.providers.contract",
        "benchmarks.providers.nli",
        "benchmarks.providers.space2",
        "benchmarks.cli",
        "benchmarks.sst2",
        "benchmarks.stsb",
    }
    code = (
        "import json, sys; import benchmarks.compare; "
        "forbidden = " + repr(forbidden) + "; "
        "print(json.dumps(sorted("
        "name for name in sys.modules "
        "if name in forbidden or name.split('.')[0] in forbidden)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=compare.ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == []

"""Paired, same-split statistical comparison of two completed runs.

This command reads raw predictions from two completed run directories, verifies
that they score identical ordered rows, recomputes both headline metrics, and
writes one deterministic comparison artifact. It never invokes a model provider
or touches the network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

from benchmarks.metrics import spearman
from benchmarks.runner import (
    ROOT,
    atomic_json,
    read_predictions,
    repository_clean,
    repository_revision,
)

SCHEMA_VERSION = 1
DEFAULT_SEED = 12345
DEFAULT_SAMPLES = 100_000
DEFAULT_CONFIDENCE = 0.95
MIN_VALID_FRACTION = 0.5
CLASSIFICATION_TYPES = {"choice", "noul"}
SUPPORTED_TYPES = {"choice", "noul", "score"}
INTERVAL_METHOD = "paired-percentile"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def rows_digest(records: list[dict[str, Any]]) -> str:
    """Deterministic digest over ordered row IDs, states, and expected labels."""
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["dataset_id"]):
        value = [record["dataset_id"], record["state"], record["expected"]]
        digest.update(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def result_type(records: list[dict[str, Any]]) -> str:
    if not records:
        raise ValueError("run contains no predictions")
    types = {record["result"]["type"] for record in records}
    if len(types) != 1:
        raise ValueError(f"run mixes result types: {sorted(types)}")
    value = types.pop()
    if value not in SUPPORTED_TYPES:
        raise ValueError(f"unsupported result type: {value!r}")
    return value


def metric_for(result: str) -> str:
    return "accuracy" if result in CLASSIFICATION_TYPES else "spearman"


def load_run(directory: Path) -> dict[str, Any]:
    directory = Path(directory).resolve()
    run_path = directory / "run.json"
    predictions_path = directory / "predictions.jsonl"
    if not run_path.exists():
        raise FileNotFoundError(f"{run_path} does not exist")
    if not predictions_path.exists():
        raise FileNotFoundError(f"{predictions_path} does not exist")
    run = json.loads(run_path.read_text())
    if run.get("status") != "complete":
        raise ValueError(f"{directory} is not a complete run (status={run.get('status')!r})")
    if run.get("evaluated") != run.get("total"):
        raise ValueError(f"{directory} is incomplete ({run.get('evaluated')} of {run.get('total')})")
    identity = run.get("identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("dataset"), dict):
        raise ValueError(f"{directory} is missing a benchmark dataset identity")
    total = run["total"]
    records = read_predictions(predictions_path)
    ids = sorted(record["dataset_id"] for record in records)
    if ids != list(range(total)):
        missing = sorted(set(range(total)) - set(ids))
        raise ValueError(f"{directory} is missing canonical rows: {missing[:5]}")
    return {
        "directory": directory,
        "run": run,
        "records": records,
        "run_path": run_path,
        "predictions_path": predictions_path,
    }


def _hits(records: list[dict[str, Any]], result: str) -> list[bool]:
    if result == "choice":
        return [record["result"]["choice"] == record["expected"] for record in records]
    return [
        (float(record["result"]["noul"]) >= 0.5) == bool(record["expected"]) for record in records
    ]


def _accuracy(records: list[dict[str, Any]], result: str) -> float:
    hits = _hits(records, result)
    return sum(hits) / len(hits)


def _spearman_score(records: list[dict[str, Any]]) -> float | None:
    expected = [float(record["expected"]) for record in records]
    predicted = [float(record["result"]["score"]) for record in records]
    return spearman(expected, predicted)


def _verify_stored_metric(run: dict[str, Any], metric: str, value: float) -> None:
    stored = run.get("metrics", {}).get(metric)
    if stored is None:
        raise ValueError(f"run does not record a {metric} metric")
    if not math.isclose(float(stored), value, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"stored {metric} {stored} does not match recomputed {value}")


def validate_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_identity = left["run"]["identity"]
    right_identity = right["run"]["identity"]
    if left_identity.get("benchmark") != right_identity.get("benchmark"):
        raise ValueError(
            f"benchmark identities differ: {left_identity.get('benchmark')!r} "
            f"versus {right_identity.get('benchmark')!r}"
        )
    left_dataset = left_identity["dataset"]
    right_dataset = right_identity["dataset"]
    if left_dataset != right_dataset:
        raise ValueError("dataset identities differ between runs")
    left_total, right_total = left["run"]["total"], right["run"]["total"]
    if left_total != right_total:
        raise ValueError(f"row counts differ: {left_total} versus {right_total}")
    if left_dataset.get("rows") is not None and left_dataset["rows"] != left_total:
        raise ValueError(f"identity row count {left_dataset['rows']} does not match {left_total} rows")
    left_result, right_result = result_type(left["records"]), result_type(right["records"])
    left_metric, right_metric = metric_for(left_result), metric_for(right_result)
    if left_metric != right_metric:
        raise ValueError(f"incompatible metrics: {left_metric} versus {right_metric}")
    left_by_id = {record["dataset_id"]: record for record in left["records"]}
    right_by_id = {record["dataset_id"]: record for record in right["records"]}
    for row_id in range(left_total):
        if row_id not in left_by_id or row_id not in right_by_id:
            raise ValueError(f"row {row_id} is missing from one of the runs")
        if left_by_id[row_id]["state"] != right_by_id[row_id]["state"]:
            raise ValueError(f"row {row_id} states differ between runs")
        if left_by_id[row_id]["expected"] != right_by_id[row_id]["expected"]:
            raise ValueError(f"row {row_id} expected values differ between runs")
    left_records = [left_by_id[row_id] for row_id in range(left_total)]
    right_records = [right_by_id[row_id] for row_id in range(left_total)]
    left_digest, right_digest = rows_digest(left_records), rows_digest(right_records)
    if left_digest != right_digest:
        raise ValueError("row digests differ between runs")
    return {
        "benchmark": left_identity["benchmark"],
        "dataset": left_dataset,
        "total": left_total,
        "rows_digest": left_digest,
        "result": left_result,
        "metric": left_metric,
        "left_records": left_records,
        "right_records": right_records,
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    position = quantile * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def percentile_interval(values: list[float], confidence: float) -> dict[str, float]:
    if not values:
        raise ValueError("cannot compute an interval from no bootstrap samples")
    ordered = sorted(values)
    alpha = (1 - confidence) / 2
    return {"lower": _percentile(ordered, alpha), "upper": _percentile(ordered, 1 - alpha)}


def bootstrap_accuracy(
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    result: str,
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> tuple[dict[str, float], int]:
    left_hits, right_hits = _hits(left_records, result), _hits(right_records, result)
    differences = [int(left) - int(right) for left, right in zip(left_hits, right_hits)]
    total = len(differences)
    generator = random.Random(seed)
    draws = [sum(generator.choices(differences, k=total)) / total for _ in range(samples)]
    return percentile_interval(draws, confidence), 0


def bootstrap_spearman(
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> tuple[dict[str, float], int]:
    expected = [float(record["expected"]) for record in left_records]
    left_predicted = [float(record["result"]["score"]) for record in left_records]
    right_predicted = [float(record["result"]["score"]) for record in right_records]
    total = len(expected)
    generator = random.Random(seed)
    valid: list[float] = []
    excluded = 0
    for _ in range(samples):
        indices = generator.choices(range(total), k=total)
        sample_expected = [expected[index] for index in indices]
        left_score = spearman(sample_expected, [left_predicted[index] for index in indices])
        right_score = spearman(sample_expected, [right_predicted[index] for index in indices])
        if left_score is None or right_score is None:
            excluded += 1
            continue
        valid.append(left_score - right_score)
    if len(valid) < samples * MIN_VALID_FRACTION:
        raise ValueError(f"too few valid bootstrap samples: {len(valid)} of {samples}")
    return percentile_interval(valid, confidence), excluded


def _outcomes(
    left_records: list[dict[str, Any]], right_records: list[dict[str, Any]], result: str
) -> dict[str, int]:
    left_hits, right_hits = _hits(left_records, result), _hits(right_records, result)
    return {
        "both_correct": sum(left and right for left, right in zip(left_hits, right_hits)),
        "left_only": sum(left and not right for left, right in zip(left_hits, right_hits)),
        "right_only": sum(right and not left for left, right in zip(left_hits, right_hits)),
        "both_wrong": sum(not left and not right for left, right in zip(left_hits, right_hits)),
    }


def _source(run: dict[str, Any], score: float | None) -> dict[str, Any]:
    return {
        "path": _display(run["directory"]),
        "run_sha256": _sha256(run["run_path"]),
        "predictions_sha256": _sha256(run["predictions_path"]),
        "identity": run["run"]["identity"],
        "score": score,
    }


def compare(
    left_directory: Path,
    right_directory: Path,
    *,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    if samples < 1:
        raise ValueError("--samples must be at least 1")
    if not 0 < confidence < 1:
        raise ValueError("--confidence must be between zero and one")
    left, right = load_run(Path(left_directory)), load_run(Path(right_directory))
    pair = validate_pair(left, right)
    metric, result = pair["metric"], pair["result"]
    left_records, right_records = pair["left_records"], pair["right_records"]

    if metric == "accuracy":
        left_score = _accuracy(left_records, result)
        right_score = _accuracy(right_records, result)
        interval, excluded = bootstrap_accuracy(
            left_records, right_records, result, samples=samples, seed=seed, confidence=confidence
        )
        outcomes: dict[str, int] | None = _outcomes(left_records, right_records, result)
    else:
        left_score = _spearman_score(left_records)
        right_score = _spearman_score(right_records)
        if left_score is None or right_score is None:
            raise ValueError("headline Spearman correlation is undefined")
        interval, excluded = bootstrap_spearman(
            left_records, right_records, samples=samples, seed=seed, confidence=confidence
        )
        outcomes = None

    _verify_stored_metric(left["run"], metric, left_score)
    _verify_stored_metric(right["run"], metric, right_score)

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": pair["benchmark"],
        "metric": metric,
        "split": pair["dataset"].get("split"),
        "rows": pair["total"],
        "row_digest": pair["rows_digest"],
        "dataset": pair["dataset"],
        "bootstrap": {
            "method": INTERVAL_METHOD,
            "confidence_level": confidence,
            "samples": samples,
            "seed": seed,
            "excluded_samples": excluded,
        },
        "left": _source(left, left_score),
        "right": _source(right, right_score),
        "difference": {
            "label": "left_minus_right",
            "value": left_score - right_score,
            "interval": interval,
        },
        "outcomes": outcomes,
        "code": {"revision": repository_revision(), "clean": repository_clean()},
    }


def default_output(benchmark: str, left_directory: Path, right_directory: Path) -> Path:
    name = f"{Path(left_directory).name}--{Path(right_directory).name}.json"
    return ROOT / "results" / benchmark / "comparisons" / name


def publish(artifact: dict[str, Any], path: Path) -> None:
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != artifact:
            raise FileExistsError(f"refusing to overwrite conflicting comparison: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, artifact)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if not 0 < args.confidence < 1:
        parser.error("--confidence must be between zero and one")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact = compare(
        args.left,
        args.right,
        samples=args.samples,
        seed=args.seed,
        confidence=args.confidence,
    )
    output = args.output or default_output(artifact["benchmark"], args.left, args.right)
    publish(artifact, output)
    print(
        json.dumps(
            {
                "benchmark": artifact["benchmark"],
                "metric": artifact["metric"],
                "rows": artifact["rows"],
                "left": artifact["left"]["score"],
                "right": artifact["right"]["score"],
                "difference": artifact["difference"]["value"],
                "interval": artifact["difference"]["interval"],
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

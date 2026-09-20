"""Run the BANKING77 benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

from benchmarks.providers import Choice, ChoiceResult, Provider, TypeSafeProvider

SCHEMA_VERSION = 1
DATASET_ID = "PolyAI/banking77"
DATASET_REVISION = "1fb62b1bb4635df59a8e1b2f2bc5e0643b2856c8"
DATASET_SPLIT = "test"
INSTRUCTION = "Classify this banking customer request by choosing the most appropriate BANKING77 intent label."
MODEL = "jev-1.13.0"
ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    if not records:
        return {"accuracy": None, "mean_confidence": None, "negative_log_loss": None, "expected_calibration_error": None}
    correct, confidences, losses = 0, [], []
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(10)]
    for record in records:
        result = record["result"]
        probabilities = result["probabilities"]
        confidence = max(probabilities.values())
        hit = result["choice"] == record["expected"]
        correct += hit
        confidences.append(confidence)
        losses.append(-math.log(max(probabilities[record["expected"]], 1e-15)))
        bins[min(int(confidence * 10), 9)].append((confidence, hit))
    ece = sum(
        len(items) / len(records) * abs(sum(c for c, _ in items) / len(items) - sum(hit for _, hit in items) / len(items))
        for items in bins
        if items
    )
    return {
        "accuracy": correct / len(records),
        "mean_confidence": sum(confidences) / len(records),
        "negative_log_loss": sum(losses) / len(records),
        "expected_calibration_error": ece,
    }


def read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path} at line {line_number}") from error
    ids = [record["dataset_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("predictions contain duplicate dataset IDs")
    return records


def identity(labels: Iterable[str]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "banking77",
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "split": DATASET_SPLIT},
        "instruction": INSTRUCTION,
        "candidates": list(labels),
        "provider": "typesafe",
        "model": MODEL,
    }


def summarize(run: dict[str, Any], records: list[dict[str, Any]], total: int, status: str) -> None:
    run.update({"updated_at": now(), "status": status, "evaluated": len(records), "total": total, "metrics": metrics(records)})
    confidences = [record["provider_metadata"].get("confidence") for record in records]
    confidences = [value for value in confidences if value is not None]
    usage: dict[str, int] = {}
    for record in records:
        for key, value in record["provider_metadata"].get("usage", {}).items():
            usage[key] = usage.get(key, 0) + value
    run["provider_statistics"] = {
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
        "usage": usage,
    }


def publication(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "identity": run["identity"],
        "dependencies": run["dependencies"],
        "repository_revision": run["repository_revision"],
        "evaluated": run["evaluated"],
        "total": run["total"],
        "metrics": run["metrics"],
        "provider_statistics": run["provider_statistics"],
    }


def publish(run: dict[str, Any], path: Path) -> None:
    value = publication(run)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise FileExistsError(f"refusing to overwrite conflicting result: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)


def repository_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def evaluate_one(provider: Provider, row_id: int, row: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    question = Choice(INSTRUCTION, labels)
    expected = labels[row["label"]]
    inference = provider.infer(row["text"], question)
    if not isinstance(inference.result, ChoiceResult):
        raise ValueError("provider returned a non-choice result")
    return {
        "dataset_id": row_id,
        "state": row["text"],
        "question": question.as_dict(),
        "expected": expected,
        "result": inference.result.as_dict(),
        "provider_metadata": inference.metadata,
    }


def run_benchmark(dataset: Any, labels: tuple[str, ...], provider: Provider, output: Path, *, limit: int | None, resume: bool, concurrency: int) -> dict[str, Any]:
    requested_identity = identity(labels)
    run_path, predictions_path = output / "run.json", output / "predictions.jsonl"
    if output.exists() and not resume:
        raise FileExistsError(f"{output} already exists; pass --resume to continue it")
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        if not run_path.exists():
            raise FileNotFoundError(f"cannot resume: {run_path} does not exist")
        run = json.loads(run_path.read_text())
        if run.get("identity") != requested_identity:
            raise ValueError("existing run identity does not match this benchmark invocation")
    else:
        run = {
            "identity": requested_identity,
            "dependencies": {"datasets": version("datasets"), "typesafe-sdk": version("typesafe-sdk")},
            "repository_revision": repository_revision(),
            "started_at": now(),
        }
        summarize(run, [], len(dataset), "partial")
        atomic_json(run_path, run)

    records = read_predictions(predictions_path)
    completed = {record["dataset_id"] for record in records}
    if not completed <= set(range(len(dataset))):
        raise ValueError("predictions contain dataset IDs outside the canonical test split")
    stop = len(dataset) if limit is None else min(limit, len(dataset))
    pending = [(row_id, dataset[row_id]) for row_id in range(stop) if row_id not in completed]
    try:
        with predictions_path.open("a") as file:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(evaluate_one, provider, row_id, row, labels): row_id for row_id, row in pending}
                failure: BaseException | None = None
                for future in as_completed(futures):
                    try:
                        record = future.result()
                    except BaseException as error:
                        failure = failure or error
                        continue
                    file.write(json.dumps(record, separators=(",", ":")) + "\n")
                    file.flush()
                    records.append(record)
                    summarize(run, records, len(dataset), "complete" if len(records) == len(dataset) else "partial")
                    atomic_json(run_path, run)
                    print(f"[{len(records)}/{stop}] example {record['dataset_id']}", flush=True)
                if failure is not None:
                    raise failure
    except BaseException:
        summarize(run, records, len(dataset), "partial" if records else "failed")
        atomic_json(run_path, run)
        raise

    if {record["dataset_id"] for record in records} == set(range(len(dataset))):
        publish(run, ROOT / "results" / "banking77" / "typesafe" / f"{MODEL}.json")
    return run


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "banking77" / "typesafe" / MODEL)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args(argv)
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split=DATASET_SPLIT)
    labels = tuple(dataset.features["label"].names)
    provider = TypeSafeProvider(MODEL)
    try:
        run = run_benchmark(dataset, labels, provider, args.output, limit=args.limit, resume=args.resume, concurrency=args.concurrency)
    finally:
        provider.close()
    print(json.dumps({"status": run["status"], "evaluated": run["evaluated"], "total": run["total"], **run["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

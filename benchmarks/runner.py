"""Shared run, resume, and publication lifecycle for benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

from benchmarks.providers import Provider

ROOT = Path(__file__).resolve().parents[1]
Metrics = Callable[[list[dict[str, Any]]], dict[str, float | None]]
Evaluate = Callable[[Provider, int, dict[str, Any]], dict[str, Any]]


def now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


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


def summarize(
    run: dict[str, Any], records: list[dict[str, Any]], total: int, status: str, metrics: Metrics
) -> None:
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
        "repository_clean": run.get("repository_clean"),
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


def repository_clean() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return not result.stdout if result.returncode == 0 else None


def run_benchmark(
    dataset: Any,
    provider: Provider,
    output: Path,
    *,
    identity: dict[str, Any],
    evaluate: Evaluate,
    metrics: Metrics,
    result_path: Path,
    limit: int | None,
    resume: bool,
    concurrency: int,
    dependencies: dict[str, str] | None = None,
) -> dict[str, Any]:
    run_path, predictions_path = output / "run.json", output / "predictions.jsonl"
    if output.exists() and not resume:
        raise FileExistsError(f"{output} already exists; pass --resume to continue it")
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        if not run_path.exists():
            raise FileNotFoundError(f"cannot resume: {run_path} does not exist")
        run = json.loads(run_path.read_text())
        if run.get("identity") != identity:
            raise ValueError("existing run identity does not match this benchmark invocation")
    else:
        run = {
            "identity": identity,
            "dependencies": dependencies or {
                "datasets": version("datasets"),
                "typesafe-sdk": version("typesafe-sdk"),
            },
            "repository_revision": repository_revision(),
            "repository_clean": repository_clean(),
            "started_at": now(),
        }
        summarize(run, [], len(dataset), "partial", metrics)
        atomic_json(run_path, run)

    records = read_predictions(predictions_path)
    completed = {record["dataset_id"] for record in records}
    if not completed <= set(range(len(dataset))):
        raise ValueError("predictions contain dataset IDs outside the canonical evaluation split")
    stop = len(dataset) if limit is None else min(limit, len(dataset))
    pending = [(row_id, dataset[row_id]) for row_id in range(stop) if row_id not in completed]
    try:
        with predictions_path.open("a") as file:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(evaluate, provider, row_id, row): row_id for row_id, row in pending}
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
                    summarize(run, records, len(dataset), "complete" if len(records) == len(dataset) else "partial", metrics)
                    atomic_json(run_path, run)
                    print(f"[{len(records)}/{stop}] example {record['dataset_id']}", flush=True)
                if failure is not None:
                    raise failure
    except BaseException:
        summarize(run, records, len(dataset), "partial" if records else "failed", metrics)
        atomic_json(run_path, run)
        raise

    if {record["dataset_id"] for record in records} == set(range(len(dataset))):
        publish(run, result_path)
    return run


def parse_run_args(description: str, default_output: Path, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    return args

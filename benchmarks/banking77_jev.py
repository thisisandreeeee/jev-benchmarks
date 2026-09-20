"""Run the BANKING77 benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

from benchmarks.metrics import classification_metrics
from benchmarks.providers import Choice, ChoiceResult, Provider, TypeSafeProvider
from benchmarks.runner import atomic_json, parse_run_args, publish, run_benchmark as run

SCHEMA_VERSION = 1
DATASET_ID = "PolyAI/banking77"
DATASET_REVISION = "1fb62b1bb4635df59a8e1b2f2bc5e0643b2856c8"
DATASET_SPLIT = "test"
INSTRUCTION = "Classify this banking customer request by choosing the most appropriate BANKING77 intent label."
MODEL = "jev-1.13.0"
ROOT = Path(__file__).resolve().parents[1]


metrics = classification_metrics


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
    return run(
        dataset,
        provider,
        output,
        identity=identity(labels),
        evaluate=lambda provider, row_id, row: evaluate_one(provider, row_id, row, labels),
        metrics=metrics,
        result_path=ROOT / "results" / "banking77" / "typesafe" / f"{MODEL}.json",
        limit=limit,
        resume=resume,
        concurrency=concurrency,
    )


def parse_args(argv: list[str] | None = None):
    return parse_run_args(__doc__ or "", ROOT / "runs" / "banking77" / "typesafe" / MODEL, argv)


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

"""Run the SST-2 sentiment benchmark on the public GLUE validation split."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from benchmarks.metrics import noul_metrics
from benchmarks.providers import Noul, NoulResult, Provider, TypeSafeProvider
from benchmarks.runner import parse_run_args, run_benchmark

SCHEMA_VERSION = 2
DATASET_ID = "nyu-mll/glue"
DATASET_CONFIG = "sst2"
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
DATASET_SPLIT = "validation"
INSTRUCTION = "Does this movie-review sentence express positive sentiment?"
MODEL = "jev-1.13.0"
ROOT = Path(__file__).resolve().parents[1]


def identity() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "sst2",
        "dataset": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
        },
        "instruction": INSTRUCTION,
        "positive_outcome": "positive",
        "provider": "typesafe",
        "model": MODEL,
    }


def evaluate_one(provider: Provider, row_id: int, row: dict[str, Any]) -> dict[str, Any]:
    question = Noul(INSTRUCTION)
    inference = provider.infer(row["sentence"], question)
    if not isinstance(inference.result, NoulResult):
        raise ValueError("provider returned a non-noul result")
    return {
        "dataset_id": row_id,
        "state": row["sentence"],
        "question": question.as_dict(),
        "expected": row["label"],
        "result": inference.result.as_dict(),
        "provider_metadata": inference.metadata,
    }


def run(
    dataset: Any,
    provider: Provider,
    output: Path,
    *,
    limit: int | None,
    resume: bool,
    concurrency: int,
) -> dict[str, Any]:
    return run_benchmark(
        dataset,
        provider,
        output,
        identity=identity(),
        evaluate=evaluate_one,
        metrics=noul_metrics,
        result_path=ROOT / "results" / "sst2" / "typesafe" / f"{MODEL}.json",
        limit=limit,
        resume=resume,
        concurrency=concurrency,
    )


def parse_args(argv: list[str] | None = None):
    return parse_run_args(__doc__ or "", ROOT / "runs" / "sst2" / "typesafe" / MODEL, argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    args = parse_args(argv)
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, revision=DATASET_REVISION, split=DATASET_SPLIT)
    provider = TypeSafeProvider(MODEL)
    try:
        result = run(dataset, provider, args.output, limit=args.limit, resume=args.resume, concurrency=args.concurrency)
    finally:
        provider.close()
    print(json.dumps({"status": result["status"], "evaluated": result["evaluated"], "total": result["total"], **result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

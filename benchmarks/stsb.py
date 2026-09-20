"""Run the STS-B similarity benchmark on the public GLUE validation split."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from benchmarks.metrics import similarity_metrics
from benchmarks.providers import Provider, Score, ScoreResult, TypeSafeProvider
from benchmarks.runner import parse_run_args, run_benchmark

SCHEMA_VERSION = 1
DATASET_ID = "nyu-mll/glue"
DATASET_CONFIG = "stsb"
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
DATASET_SPLIT = "validation"
INSTRUCTION = "Rate the semantic similarity of the two sentences using the rubric."
RUBRIC = (
    "The sentences are completely dissimilar.",
    "The sentences are on the same topic but are not equivalent.",
    "The sentences share some details but are not equivalent.",
    "The sentences are roughly equivalent, but important information differs.",
    "The sentences are mostly equivalent, with only minor differences.",
    "The sentences are completely equivalent in meaning.",
)
MODEL = "jev-1.13.0"
ROOT = Path(__file__).resolve().parents[1]


def identity() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "stsb",
        "dataset": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
        },
        "instruction": INSTRUCTION,
        "rubric": list(RUBRIC),
        "provider": "typesafe",
        "model": MODEL,
    }


def evaluate_one(provider: Provider, row_id: int, row: dict[str, Any]) -> dict[str, Any]:
    state = f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}"
    question = Score(INSTRUCTION, RUBRIC)
    inference = provider.infer(state, question)
    if not isinstance(inference.result, ScoreResult):
        raise ValueError("provider returned a non-score result")
    return {
        "dataset_id": row_id,
        "state": state,
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
        metrics=similarity_metrics,
        result_path=ROOT / "results" / "stsb" / "typesafe" / f"{MODEL}.json",
        limit=limit,
        resume=resume,
        concurrency=concurrency,
    )


def parse_args(argv: list[str] | None = None):
    return parse_run_args(__doc__ or "", ROOT / "runs" / "stsb" / "typesafe" / MODEL, argv)


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

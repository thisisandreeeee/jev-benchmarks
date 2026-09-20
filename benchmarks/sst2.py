"""Run SST-2 validation evaluations with DistilBERT or TypeSafe Jev."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from benchmarks.metrics import noul_metrics
from benchmarks.providers import Noul, NoulResult, Provider, TransformersNoulProvider, TypeSafeProvider
from benchmarks.runner import run_benchmark

JEV_SCHEMA_VERSION = 2
HUGGINGFACE_SCHEMA_VERSION = 1
BENCHMARK = "sst2"
DATASET_ID = "nyu-mll/glue"
DATASET_CONFIG = "sst2"
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
DATASET_SPLIT = "validation"
INSTRUCTION = "Does this movie-review sentence express positive sentiment?"
JEV_MODEL = "jev-1.13.0"
MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MODEL_REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"
MODEL_CARD = f"https://huggingface.co/{MODEL}"
POSITIVE_LABEL = "POSITIVE"
SLUG = "distilbert-base-uncased-finetuned-sst-2-english"
ROOT = Path(__file__).resolve().parents[1]


def identity(provider: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "benchmark": BENCHMARK,
        "dataset": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
        },
        "instruction": INSTRUCTION,
        "positive_outcome": "positive",
    }
    if provider == "typesafe":
        value.update({"schema_version": JEV_SCHEMA_VERSION, "provider": "typesafe", "model": JEV_MODEL})
    elif provider == "huggingface":
        value.update(
            {
                "schema_version": HUGGINGFACE_SCHEMA_VERSION,
                "provider": "huggingface-local",
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "model_card": MODEL_CARD,
                "positive_label": POSITIVE_LABEL,
            }
        )
    else:
        raise ValueError(f"unsupported provider: {provider}")
    return value


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
    provider_name: str,
    output: Path,
    *,
    limit: int | None,
    resume: bool,
    concurrency: int,
) -> dict[str, Any]:
    if provider_name == "typesafe":
        slug = JEV_MODEL
        dependencies = {"datasets": version("datasets"), "typesafe-sdk": version("typesafe-sdk")}
    elif provider_name == "huggingface":
        slug = SLUG
        dependencies = {
            "datasets": version("datasets"),
            "torch": version("torch"),
            "transformers": version("transformers"),
        }
    else:
        raise ValueError(f"unsupported provider: {provider_name}")
    return run_benchmark(
        dataset,
        provider,
        output,
        identity=identity(provider_name),
        evaluate=evaluate_one,
        metrics=noul_metrics,
        result_path=ROOT / "results" / BENCHMARK / provider_name / f"{slug}.json",
        limit=limit,
        resume=resume,
        concurrency=concurrency,
        dependencies=dependencies,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("typesafe", "huggingface"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.output is None:
        slug = JEV_MODEL if args.provider == "typesafe" else SLUG
        args.output = ROOT / "runs" / BENCHMARK / args.provider / slug
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, revision=DATASET_REVISION, split=DATASET_SPLIT)
    if args.provider == "typesafe":
        load_dotenv(ROOT / ".env")
        provider: Provider = TypeSafeProvider(JEV_MODEL)
    else:
        provider = TransformersNoulProvider(MODEL, MODEL_REVISION, POSITIVE_LABEL)
    try:
        result = run(
            dataset,
            provider,
            args.provider,
            args.output,
            limit=args.limit,
            resume=args.resume,
            concurrency=args.concurrency,
        )
    finally:
        close = getattr(provider, "close", None)
        if close is not None:
            close()
    print(
        json.dumps(
            {"status": result["status"], "evaluated": result["evaluated"], "total": result["total"], **result["metrics"]},
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

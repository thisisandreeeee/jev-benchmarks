"""Run the supervised DistilBERT SST-2 baseline on the GLUE validation split."""

from __future__ import annotations

import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from benchmarks.metrics import noul_metrics
from benchmarks.providers import Provider, TransformersNoulProvider
from benchmarks.runner import parse_run_args, run_benchmark
from benchmarks.sst2_jev import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    INSTRUCTION,
    evaluate_one,
)

SCHEMA_VERSION = 1
MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MODEL_REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"
MODEL_CARD = f"https://huggingface.co/{MODEL}"
POSITIVE_LABEL = "POSITIVE"
ROOT = Path(__file__).resolve().parents[1]
SLUG = "distilbert-base-uncased-finetuned-sst-2-english"


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
        "provider": "huggingface-local",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "model_card": MODEL_CARD,
        "positive_label": POSITIVE_LABEL,
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
        result_path=ROOT / "results" / "sst2" / "huggingface" / f"{SLUG}.json",
        limit=limit,
        resume=resume,
        concurrency=concurrency,
        dependencies={
            "datasets": version("datasets"),
            "torch": version("torch"),
            "transformers": version("transformers"),
        },
    )


def parse_args(argv: list[str] | None = None):
    default = ROOT / "runs" / "sst2" / "huggingface" / SLUG
    return parse_run_args(__doc__ or "", default, argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, revision=DATASET_REVISION, split=DATASET_SPLIT)
    provider = TransformersNoulProvider(MODEL, MODEL_REVISION, POSITIVE_LABEL)
    result = run(
        dataset,
        provider,
        args.output,
        limit=args.limit,
        resume=args.resume,
        concurrency=args.concurrency,
    )
    print(json.dumps({"status": result["status"], "evaluated": result["evaluated"], "total": result["total"], **result["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

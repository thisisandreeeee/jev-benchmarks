"""Run SST-2 validation evaluations with RoBERTa-large or TypeSafe Jev."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from benchmarks import cli
from benchmarks.metrics import noul_metrics
from benchmarks.providers import (
    NliZeroShotProvider,
    Noul,
    NoulResult,
    Provider,
    TransformersNoulProvider,
    TypeSafeProvider,
)
from benchmarks.providers import nli_manifest as nli
from benchmarks.runner import add_run_arguments, run_benchmark, validate_run_arguments

JEV_SCHEMA_VERSION = 2
HUGGINGFACE_SCHEMA_VERSION = 1
NLI_SCHEMA_VERSION = 1
BENCHMARK = "sst2"
DATASET_ID = "nyu-mll/glue"
DATASET_CONFIG = "sst2"
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
DATASET_SPLIT = "validation"
EXPECTED_ROWS = 872
ROW_DIGEST = "b85e9210b8507d7c79e2b9d5220d4c4152e6f9c19349504318523acc74ec8c3e"
INSTRUCTION = "Does this movie-review sentence express positive sentiment?"
JEV_MODEL = "jev-1.13.0"
MODEL = "philschmid/roberta-large-sst2"
MODEL_REVISION = "7d2599d698b7a805b6831e15e830e60a0b07bdb4"
MODEL_CARD = f"https://huggingface.co/{MODEL}"
POSITIVE_LABEL = "positive"
SLUG = "roberta-large-sst2"
ROOT = Path(__file__).resolve().parents[1]

PROVIDERS = {
    "typesafe": cli.ProviderBinding(JEV_MODEL, "typesafe"),
    "huggingface": cli.ProviderBinding(SLUG, "transformers"),
    "nli": cli.ProviderBinding(nli.SLUG, "transformers", concurrency=1),
}


def row_digest(dataset: Any) -> str:
    digest = hashlib.sha256()
    for row in dataset:
        digest.update(json.dumps([row["sentence"], row["label"]], separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def validate_dataset(dataset: Any) -> None:
    if len(dataset) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} SST-2 validation rows, got {len(dataset)}")
    actual = row_digest(dataset)
    if actual != ROW_DIGEST:
        raise ValueError(f"SST-2 validation row digest mismatch: {actual}")


def identity(provider: str, nli_manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "benchmark": BENCHMARK,
        "dataset": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "rows": EXPECTED_ROWS,
            "row_digest": ROW_DIGEST,
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
    elif provider == "nli":
        value["schema_version"] = NLI_SCHEMA_VERSION
        value.update(nli.provider_identity(nli_manifest or nli.load_manifest(), "sst2"))
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
    nli_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_dataset(dataset)
    binding = PROVIDERS[provider_name]
    return run_benchmark(
        dataset,
        provider,
        output,
        identity=identity(provider_name, nli_manifest),
        evaluate=evaluate_one,
        metrics=noul_metrics,
        result_path=ROOT / "results" / BENCHMARK / f"{provider_name}-{binding.slug}.json",
        limit=limit,
        resume=resume,
        concurrency=concurrency,
        dependencies=binding.dependencies(),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=tuple(PROVIDERS), required=True)
    add_run_arguments(parser)
    parser.add_argument("--nli-manifest", type=Path, default=nli.DEFAULT_MANIFEST)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    validate_run_arguments(parser, args)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    binding = PROVIDERS[args.provider]
    if binding.concurrency is not None and args.concurrency != binding.concurrency:
        parser.error(f"{args.provider} requires --concurrency {binding.concurrency}")
    if args.output is None:
        args.output = ROOT / "runs" / BENCHMARK / f"{args.provider}-{binding.slug}"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, revision=DATASET_REVISION, split=DATASET_SPLIT)
    nli_manifest: dict[str, Any] | None = None
    if args.provider == "typesafe":
        load_dotenv(ROOT / ".env")
        provider: Provider = TypeSafeProvider(JEV_MODEL)
    elif args.provider == "huggingface":
        provider = TransformersNoulProvider(MODEL, MODEL_REVISION, POSITIVE_LABEL)
    else:
        manifest = nli.load_manifest(args.nli_manifest)
        nli_manifest = manifest
        entry = nli.benchmark_entry(manifest, BENCHMARK)
        provider = NliZeroShotProvider(
            model=manifest["model"],
            revision=manifest["revision"],
            labels=tuple(entry["labels"]),
            verbalization=entry["verbalization"],
            template=manifest["hypothesis_template"],
            positive_label=POSITIVE_LABEL,
            max_length=manifest["max_length"],
            device=args.device,
            batch_size=args.batch_size,
        )
    return cli.finish(
        provider,
        lambda: run(
            dataset,
            provider,
            args.provider,
            args.output,
            limit=args.limit,
            resume=args.resume,
            concurrency=args.concurrency,
            nli_manifest=nli_manifest,
        ),
        total=True,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

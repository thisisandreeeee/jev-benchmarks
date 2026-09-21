"""Shared lifecycle for SPACE-2 intent benchmarks and their Jev controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from benchmarks.metrics import classification_metrics
from benchmarks.providers import Choice, ChoiceResult, Provider, Space2Provider, TypeSafeProvider
from benchmarks.runner import add_run_arguments, run_benchmark, validate_run_arguments
from benchmarks.space2_release import (
    author_rows,
    load_manifest,
    validate_alignment,
    validate_reference_predictions,
    validate_release,
    validate_smoke,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPACE2_DIR = ROOT / ".cache" / "space2"
JEV_MODEL = "jev-1.13.0"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IntentConfig:
    benchmark: str
    dataset_id: str
    dataset_revision: str
    dataset_split: str
    expected_rows: int
    expected_labels: int
    row_digest: str
    instruction: str
    space2_model: str
    manifest: Path
    author_prefix: str
    dataset_config: str | None = None
    dataset_details: dict[str, Any] = field(default_factory=dict)
    casefold_author_text: bool = False


Loader = Callable[[Any, tuple[str, ...]], tuple[list[dict[str, str]], tuple[str, ...]]]


def row_digest(rows: Any) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps([row["text"], row["label"]], ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def validate_dataset(rows: Any, config: IntentConfig) -> None:
    if len(rows) != config.expected_rows:
        raise ValueError(f"expected {config.expected_rows} {config.benchmark} test rows, got {len(rows)}")
    actual = row_digest(rows)
    if actual != config.row_digest:
        raise ValueError(f"{config.benchmark} test row digest mismatch: {actual}")


def dataset_identity(config: IntentConfig) -> dict[str, Any]:
    return {
        "id": config.dataset_id,
        "revision": config.dataset_revision,
        "split": config.dataset_split,
        "rows": config.expected_rows,
        "row_digest": config.row_digest,
        **config.dataset_details,
    }


def identity(
    provider: str, labels: tuple[str, ...], config: IntentConfig, manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": config.benchmark,
        "dataset": dataset_identity(config),
        "instruction": config.instruction,
        "candidates": list(labels),
    }
    if provider == "typesafe":
        value.update({"provider": "typesafe", "model": JEV_MODEL})
    elif provider == "space-2" and manifest is not None:
        value.update(
            {
                "provider": "space-2-local",
                "model": config.space2_model,
                "repository": manifest["repository"],
                "paper": manifest["paper"],
                "checkpoint": manifest["checkpoint"],
                "preprocessing": manifest["preprocessing"],
                "label_mapping": manifest["label_mapping"],
                "manifest_sha256": manifest["manifest_sha256"],
            }
        )
    else:
        raise ValueError(f"unsupported provider: {provider}")
    return value


def evaluate_one(provider: Provider, row_id: int, row: dict[str, str], labels: tuple[str, ...], instruction: str):
    question = Choice(instruction, labels)
    inference = provider.infer(row["text"], question)
    if not isinstance(inference.result, ChoiceResult):
        raise ValueError("provider returned a non-choice result")
    return {
        "dataset_id": row_id,
        "state": row["text"],
        "question": question.as_dict(),
        "expected": row["label"],
        "result": inference.result.as_dict(),
        "provider_metadata": inference.metadata,
    }


def run(
    rows: Any,
    labels: tuple[str, ...],
    provider: Provider,
    provider_name: str,
    output: Path,
    config: IntentConfig,
    *,
    manifest: dict[str, Any] | None,
    limit: int | None,
    resume: bool,
    concurrency: int,
) -> dict[str, Any]:
    validate_dataset(rows, config)
    prepare = getattr(provider, "prepare", None)
    if prepare is not None:
        if output.exists() and not resume:
            raise FileExistsError(f"{output} already exists; pass --resume to continue it")
        stop = len(rows) if limit is None else min(limit, len(rows))
        completed: set[int] = set()
        predictions = output / "predictions.jsonl"
        if resume and predictions.exists():
            completed = {json.loads(line)["dataset_id"] for line in predictions.read_text().splitlines()}
        prepare([rows[index]["text"] for index in range(stop) if index not in completed])
    slug = JEV_MODEL if provider_name == "typesafe" else config.space2_model
    dependencies = {"datasets": version("datasets")}
    if provider_name == "typesafe":
        dependencies["typesafe-sdk"] = version("typesafe-sdk")
    else:
        dependencies.update({"torch": version("torch"), "transformers": version("transformers")})
    return run_benchmark(
        rows,
        provider,
        output,
        identity=identity(provider_name, labels, config, manifest),
        evaluate=lambda provider, row_id, row: evaluate_one(provider, row_id, row, labels, config.instruction),
        metrics=classification_metrics,
        result_path=ROOT / "results" / config.benchmark / provider_name / f"{slug}.json",
        limit=limit,
        resume=resume,
        concurrency=concurrency,
        dependencies=dependencies,
    )


def parse_args(config: IntentConfig, argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run {config.benchmark} with SPACE-2 or TypeSafe Jev.")
    parser.add_argument("--provider", choices=("typesafe", "space-2"), required=True)
    add_run_arguments(parser)
    parser.add_argument("--manifest", type=Path, default=config.manifest)
    parser.add_argument("--space2-dir", type=Path, default=DEFAULT_SPACE2_DIR)
    args = parser.parse_args(argv)
    validate_run_arguments(parser, args)
    if args.provider == "space-2" and args.concurrency != 1:
        parser.error("SPACE-2 requires --concurrency 1")
    if args.output is None:
        slug = JEV_MODEL if args.provider == "typesafe" else config.space2_model
        args.output = ROOT / "runs" / config.benchmark / args.provider / slug
    return args


def main(config: IntentConfig, loader: Loader, argv: list[str] | None = None) -> int:
    args = parse_args(config, argv)
    from datasets import load_dataset

    manifest = load_manifest(args.manifest, config.expected_labels)
    dataset = load_dataset(
        config.dataset_id,
        config.dataset_config,
        revision=config.dataset_revision,
        split=config.dataset_split,
    )
    rows, labels = loader(dataset, tuple(manifest["label_mapping"]))
    validate_dataset(rows, config)
    if args.provider == "typesafe":
        load_dotenv(ROOT / ".env")
        provider: Provider = TypeSafeProvider(JEV_MODEL)
        run_manifest = None
    else:
        paths = validate_release(args.space2_dir, manifest)
        mapping = tuple(manifest["label_mapping"])
        validate_alignment(rows, labels, paths["author_test"], mapping, config)
        reference = validate_reference_predictions(
            paths["reference_predictions"], mapping, config.expected_rows, manifest["reference_accuracy"]
        )
        provider = Space2Provider(
            str(paths["checkpoint"]), str(paths["vocab"]), mapping, manifest["checkpoint"]["sha256"]
        )
        validate_smoke(provider, rows, labels, paths["author_test"], reference, mapping, config)
        run_manifest = manifest
    try:
        result = run(
            rows,
            labels,
            provider,
            args.provider,
            args.output,
            config,
            manifest=run_manifest,
            limit=args.limit,
            resume=args.resume,
            concurrency=args.concurrency,
        )
    finally:
        close = getattr(provider, "close", None)
        if close is not None:
            close()
    print(json.dumps({"status": result["status"], "evaluated": result["evaluated"], **result["metrics"]}, indent=2))
    return 0

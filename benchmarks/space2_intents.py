"""Shared lifecycle for SPACE-2 intent benchmarks and their Jev controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from benchmarks.metrics import classification_metrics
from benchmarks.providers import Choice, ChoiceResult, Provider, Space2Provider, TypeSafeProvider
from benchmarks.runner import run_benchmark

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path, expected_labels: int) -> dict[str, Any]:
    value = json.loads(path.read_text())
    mapping = value.get("label_mapping")
    if value.get("schema_version") != 1 or not isinstance(mapping, list) or len(mapping) != expected_labels:
        raise ValueError("invalid SPACE-2 manifest")
    if len(set(mapping)) != expected_labels or any(not isinstance(label, str) for label in mapping):
        raise ValueError(f"SPACE-2 label mapping must contain {expected_labels} unique labels")
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("SPACE-2 manifest does not define release files")
    value["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return value


def validate_release(root: Path, manifest: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    resolved_root = root.resolve()
    for name, item in manifest["files"].items():
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError(f"SPACE-2 manifest path escapes release directory: {item['path']}")
        if not path.is_file():
            raise FileNotFoundError(f"missing SPACE-2 release file: {path}")
        if path.stat().st_size != item["size"] or _sha256(path) != item["sha256"]:
            raise ValueError(f"SPACE-2 release digest mismatch: {path}")
        paths[name] = path
    return paths


def author_rows(path: Path, prefix: str) -> tuple[dict[int, tuple[str, str, int]], dict[int, int]]:
    raw = json.loads(path.read_text())
    rows: dict[int, tuple[str, str, int]] = {}
    positions: dict[int, int] = {}
    for position, (key, dialog) in enumerate(raw.items()):
        suffix = key.removeprefix(prefix)
        if not key.startswith(prefix) or not suffix.isdigit() or len(dialog.get("turns", ())) != 1:
            raise ValueError("invalid SPACE-2 intent row")
        row_id = int(suffix)
        turn = dialog["turns"][0]
        frame = turn["label"].get("DEFAULT_DOMAIN", {})
        if len(frame) != 1 or row_id in rows:
            raise ValueError("invalid or duplicate SPACE-2 intent row")
        rows[row_id] = (turn["text"], next(iter(frame)), turn["extra_info"]["intent_label"])
        positions[row_id] = position
    return rows, positions


def validate_alignment(
    rows: Any,
    labels: tuple[str, ...],
    author_data: Path,
    mapping: tuple[str, ...],
    config: IntentConfig,
) -> None:
    author, _ = author_rows(author_data, config.author_prefix)
    if set(author) != set(range(len(rows))) or set(mapping) != set(labels):
        raise ValueError(f"SPACE-2 data does not cover the canonical {config.benchmark} split and labels")
    for row_id, row in enumerate(rows):
        text, label, index = author[row_id]
        texts_match = text.casefold() == row["text"].casefold() if config.casefold_author_text else text == row["text"]
        if not texts_match or label != row["label"] or not 0 <= index < len(mapping) or mapping[index] != label:
            raise ValueError(f"SPACE-2 data mismatch at canonical row {row_id}")


def validate_reference_predictions(
    path: Path, mapping: tuple[str, ...], expected_rows: int, expected_accuracy: float
) -> list[list[float]]:
    value = json.loads(path.read_text())
    predictions = value.get("pred_labels")
    if not isinstance(predictions, list) or len(predictions) != expected_rows:
        raise ValueError("invalid SPACE-2 reference predictions")
    for prediction in predictions:
        if (
            not isinstance(prediction, list)
            or len(prediction) != len(mapping)
            or any(not isinstance(item, (int, float)) or not math.isfinite(item) for item in prediction)
            or not math.isclose(sum(prediction), 1.0, abs_tol=1e-5)
        ):
            raise ValueError("invalid SPACE-2 reference probability distribution")
    if not math.isclose(float(value.get("accuracy")), expected_accuracy, abs_tol=1e-12):
        raise ValueError("unexpected SPACE-2 reference accuracy")
    return predictions


def validate_smoke(
    provider: Provider,
    rows: Any,
    labels: tuple[str, ...],
    author_data: Path,
    reference: list[list[float]],
    mapping: tuple[str, ...],
    config: IntentConfig,
) -> None:
    _, positions = author_rows(author_data, config.author_prefix)
    question = Choice(config.instruction, labels)
    for row_id in (0, config.expected_rows // 2, config.expected_rows - 1):
        result = provider.infer(rows[row_id]["text"], question).result
        if not isinstance(result, ChoiceResult):
            raise ValueError("SPACE-2 smoke inference returned a non-choice result")
        actual = [result.probabilities[label] for label in mapping]
        expected = reference[positions[row_id]]
        if (
            max(range(len(actual)), key=actual.__getitem__) != max(range(len(expected)), key=expected.__getitem__)
            or max(abs(left - right) for left, right in zip(actual, expected)) > 1e-4
        ):
            raise ValueError(f"SPACE-2 compatibility check failed at row {row_id}")


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
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path, default=config.manifest)
    parser.add_argument("--space2-dir", type=Path, default=DEFAULT_SPACE2_DIR)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
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

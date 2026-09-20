"""Run BANKING77 test evaluations with SPACE-2 or TypeSafe Jev."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from benchmarks.metrics import classification_metrics
from benchmarks.providers import Choice, ChoiceResult, Provider, Space2Provider, TypeSafeProvider
from benchmarks.runner import run_benchmark

SCHEMA_VERSION = 1
BENCHMARK = "banking77"
DATASET_ID = "PolyAI/banking77"
DATASET_REVISION = "1fb62b1bb4635df59a8e1b2f2bc5e0643b2856c8"
DATASET_SPLIT = "test"
EXPECTED_ROWS = 3_080
ROW_DIGEST = "e331bc3fa83d880409f6044836f709ae70de4193a4186f297385bbd6e5527808"
INSTRUCTION = "Classify this banking customer request by choosing the most appropriate BANKING77 intent label."
JEV_MODEL = "jev-1.13.0"
SPACE2_MODEL = "state_epoch_51"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests" / "space2-banking77.json"
DEFAULT_SPACE2_DIR = ROOT / ".cache" / "space2"


def row_digest(dataset: Any, labels: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for row in dataset:
        value = [row["text"], labels[row["label"]]]
        digest.update(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def validate_dataset(dataset: Any, labels: tuple[str, ...]) -> None:
    if len(dataset) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} BANKING77 test rows, got {len(dataset)}")
    actual = row_digest(dataset, labels)
    if actual != ROW_DIGEST:
        raise ValueError(f"BANKING77 test row digest mismatch: {actual}")


def dataset_identity() -> dict[str, Any]:
    return {
        "id": DATASET_ID,
        "revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
        "rows": EXPECTED_ROWS,
        "row_digest": ROW_DIGEST,
    }


def identity(provider: str, labels: tuple[str, ...], manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK,
        "dataset": dataset_identity(),
        "instruction": INSTRUCTION,
        "candidates": list(labels),
    }
    if provider == "typesafe":
        value.update({"provider": "typesafe", "model": JEV_MODEL})
    elif provider == "space-2" and manifest is not None:
        value.update(
            {
                "provider": "space-2-local",
                "model": SPACE2_MODEL,
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


def evaluate_one(provider: Provider, row_id: int, row: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    question = Choice(INSTRUCTION, labels)
    inference = provider.infer(row["text"], question)
    if not isinstance(inference.result, ChoiceResult):
        raise ValueError("provider returned a non-choice result")
    return {
        "dataset_id": row_id,
        "state": row["text"],
        "question": question.as_dict(),
        "expected": labels[row["label"]],
        "result": inference.result.as_dict(),
        "provider_metadata": inference.metadata,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    mapping = value.get("label_mapping")
    if value.get("schema_version") != 1 or not isinstance(mapping, list) or len(mapping) != 77:
        raise ValueError("invalid SPACE-2 manifest")
    if len(set(mapping)) != 77 or any(not isinstance(label, str) for label in mapping):
        raise ValueError("SPACE-2 label mapping must contain 77 unique labels")
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("SPACE-2 manifest does not define release files")
    raw = path.read_bytes()
    value["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
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


def _author_rows(path: Path) -> dict[int, tuple[str, str, int]]:
    raw = json.loads(path.read_text())
    rows: dict[int, tuple[str, str, int]] = {}
    for key, dialog in raw.items():
        if not key.startswith("banking-") or not key[8:].isdigit() or len(dialog.get("turns", ())) != 1:
            raise ValueError("invalid SPACE-2 BANKING77 row")
        row_id = int(key[8:])
        turn = dialog["turns"][0]
        frame = turn["label"].get("DEFAULT_DOMAIN", {})
        if len(frame) != 1 or row_id in rows:
            raise ValueError("invalid or duplicate SPACE-2 BANKING77 row")
        rows[row_id] = (turn["text"], next(iter(frame)), turn["extra_info"]["intent_label"])
    return rows


def validate_alignment(dataset: Any, labels: tuple[str, ...], author_data: Path, mapping: tuple[str, ...]) -> None:
    rows = _author_rows(author_data)
    if set(rows) != set(range(len(dataset))) or set(mapping) != set(labels):
        raise ValueError("SPACE-2 data does not cover the canonical BANKING77 split and labels")
    for row_id, row in enumerate(dataset):
        text, label, index = rows[row_id]
        expected = labels[row["label"]]
        if text != row["text"] or label != expected or not 0 <= index < len(mapping) or mapping[index] != expected:
            raise ValueError(f"SPACE-2 data mismatch at canonical row {row_id}")


def validate_reference_predictions(path: Path, mapping: tuple[str, ...], expected_accuracy: float) -> list[list[float]]:
    value = json.loads(path.read_text())
    predictions = value.get("pred_labels")
    if not isinstance(predictions, list) or len(predictions) != EXPECTED_ROWS:
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
    labels: tuple[str, ...],
    author_data: Path,
    reference: list[list[float]],
    mapping: tuple[str, ...],
) -> None:
    raw = json.loads(author_data.read_text())
    positions = {int(key[8:]): position for position, key in enumerate(raw)}
    rows = _author_rows(author_data)
    question = Choice(INSTRUCTION, labels)
    for row_id in (0, EXPECTED_ROWS // 2, EXPECTED_ROWS - 1):
        result = provider.infer(rows[row_id][0], question).result
        if not isinstance(result, ChoiceResult):
            raise ValueError("SPACE-2 smoke inference returned a non-choice result")
        actual = [result.probabilities[label] for label in mapping]
        expected = reference[positions[row_id]]
        if max(abs(left - right) for left, right in zip(actual, expected)) > 1e-5:
            raise ValueError(f"SPACE-2 compatibility check failed at row {row_id}")


def run(
    dataset: Any,
    labels: tuple[str, ...],
    provider: Provider,
    provider_name: str,
    output: Path,
    *,
    manifest: dict[str, Any] | None,
    limit: int | None,
    resume: bool,
    concurrency: int,
) -> dict[str, Any]:
    validate_dataset(dataset, labels)
    prepare = getattr(provider, "prepare", None)
    if prepare is not None:
        if output.exists() and not resume:
            raise FileExistsError(f"{output} already exists; pass --resume to continue it")
        stop = len(dataset) if limit is None else min(limit, len(dataset))
        completed: set[int] = set()
        predictions = output / "predictions.jsonl"
        if resume and predictions.exists():
            completed = {json.loads(line)["dataset_id"] for line in predictions.read_text().splitlines()}
        prepare([dataset[index]["text"] for index in range(stop) if index not in completed])
    slug = JEV_MODEL if provider_name == "typesafe" else SPACE2_MODEL
    dependencies = {"datasets": version("datasets")}
    if provider_name == "typesafe":
        dependencies["typesafe-sdk"] = version("typesafe-sdk")
    else:
        dependencies.update({"torch": version("torch"), "transformers": version("transformers")})
    return run_benchmark(
        dataset,
        provider,
        output,
        identity=identity(provider_name, labels, manifest),
        evaluate=lambda provider, row_id, row: evaluate_one(provider, row_id, row, labels),
        metrics=classification_metrics,
        result_path=ROOT / "results" / BENCHMARK / provider_name / f"{slug}.json",
        limit=limit,
        resume=resume,
        concurrency=concurrency,
        dependencies=dependencies,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("typesafe", "space-2"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--space2-dir", type=Path, default=DEFAULT_SPACE2_DIR)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    if args.provider == "space-2" and args.concurrency != 1:
        parser.error("SPACE-2 requires --concurrency 1")
    if args.output is None:
        slug = JEV_MODEL if args.provider == "typesafe" else SPACE2_MODEL
        args.output = ROOT / "runs" / BENCHMARK / args.provider / slug
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split=DATASET_SPLIT)
    labels = tuple(dataset.features["label"].names)
    manifest: dict[str, Any] | None = None
    if args.provider == "typesafe":
        load_dotenv(ROOT / ".env")
        provider: Provider = TypeSafeProvider(JEV_MODEL)
    else:
        manifest = load_manifest(args.manifest)
        paths = validate_release(args.space2_dir, manifest)
        mapping = tuple(manifest["label_mapping"])
        validate_alignment(dataset, labels, paths["author_test"], mapping)
        reference = validate_reference_predictions(
            paths["reference_predictions"], mapping, manifest["reference_accuracy"]
        )
        provider = Space2Provider(
            str(paths["checkpoint"]), str(paths["vocab"]), mapping, manifest["checkpoint"]["sha256"]
        )
        validate_smoke(provider, labels, paths["author_test"], reference, mapping)
    try:
        result = run(
            dataset,
            labels,
            provider,
            args.provider,
            args.output,
            manifest=manifest,
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

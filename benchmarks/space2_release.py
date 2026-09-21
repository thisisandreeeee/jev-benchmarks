"""Validation for the pinned SPACE-2 release artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from benchmarks.providers import Choice, ChoiceResult


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
    config: Any,
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
    provider: Any,
    rows: Any,
    labels: tuple[str, ...],
    author_data: Path,
    reference: list[list[float]],
    mapping: tuple[str, ...],
    config: Any,
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

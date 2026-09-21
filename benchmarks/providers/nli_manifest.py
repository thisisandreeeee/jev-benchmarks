"""Frozen configuration and validation for the local NLI zero-shot provider."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "manifests" / "nli-labels.json"
SLUG = "deberta-v3-large-zeroshot-v2.0"
EXPECTED_NORMALIZATION = "softmax_over_candidates_of_entailment_logits"


def verbalize_label(label: str) -> str:
    """Turn a canonical snake_case label into the words used in a hypothesis."""
    return label.replace("_", " ").strip().lower().strip("?.!,").strip()


def verbalization_digest(verbalization: dict[str, str]) -> str:
    blob = json.dumps(verbalization, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest = json.loads((path or DEFAULT_MANIFEST).read_text())
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("benchmarks"), dict):
        raise ValueError("invalid zero-shot manifest")
    if manifest.get("normalization") != EXPECTED_NORMALIZATION:
        raise ValueError("unexpected zero-shot normalization")
    if manifest.get("multi_label") is not False:
        raise ValueError("zero-shot manifest must disable multi_label")
    digests = manifest.get("verbalization_sha256")
    if not isinstance(digests, dict):
        raise ValueError("zero-shot manifest is missing verbalization digests")
    for name, entry in manifest["benchmarks"].items():
        if not isinstance(entry, dict):
            raise ValueError(f"invalid zero-shot manifest entry for {name}")
        labels, verbalization = entry.get("labels"), entry.get("verbalization")
        if not isinstance(labels, list) or not labels or not all(isinstance(label, str) for label in labels):
            raise ValueError(f"zero-shot manifest has no labels for {name}")
        if len(labels) != len(set(labels)) or not isinstance(verbalization, dict):
            raise ValueError(f"zero-shot manifest has invalid labels for {name}")
        if set(labels) != set(verbalization):
            raise ValueError(f"zero-shot verbalization does not cover {name}")
        if verbalization_digest(verbalization) != digests.get(name):
            raise ValueError(f"zero-shot verbalization digest mismatch for {name}")
    return manifest


def benchmark_entry(manifest: dict[str, Any], benchmark: str) -> dict[str, Any]:
    try:
        return manifest["benchmarks"][benchmark]
    except KeyError as error:
        raise ValueError(f"zero-shot manifest does not define {benchmark}") from error


def validate_labels(entry: dict[str, Any], labels: tuple[str, ...]) -> None:
    if set(labels) != set(entry["labels"]):
        raise ValueError("zero-shot verbalization does not match the canonical label set")


def provider_identity(manifest: dict[str, Any], benchmark: str) -> dict[str, Any]:
    benchmark_entry(manifest, benchmark)
    return {
        "provider": manifest["provider"],
        "model": manifest["model"],
        "model_revision": manifest["revision"],
        "model_card": manifest["model_card"],
        "license": manifest["license"],
        "hypothesis_template": manifest["hypothesis_template"],
        "entailment_label": manifest["entailment_label"],
        "normalization": manifest["normalization"],
        "max_length": manifest["max_length"],
        "verbalization_sha256": manifest["verbalization_sha256"][benchmark],
    }

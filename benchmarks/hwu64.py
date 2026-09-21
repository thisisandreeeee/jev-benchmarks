"""Run HWU64 test evaluations with SPACE-2 or TypeSafe Jev."""

from __future__ import annotations

from typing import Any

from benchmarks.space2_intents import ROOT, IntentConfig, main as run_main

CONFIG = IntentConfig(
    benchmark="hwu64",
    dataset_id="DeepPavlov/hwu64",
    dataset_config="default",
    dataset_revision="0dd289ccdeb185ec065d1ebcf5de1c443cd1620f",
    dataset_split="test",
    expected_rows=1_076,
    expected_labels=64,
    row_digest="a18555d600b1a85d49013ca4bd12171c21361fa4756aaf4040785848c9d345ec",
    instruction="Classify this request by choosing the most appropriate HWU64 intent label.",
    space2_model="state_epoch_25",
    manifest=ROOT / "manifests" / "space2-hwu64.json",
    author_prefix="hwu-",
    dataset_details={"config": "default", "text_normalization": "canonical source; case-insensitive author check"},
    casefold_author_text=True,
)


def load_rows(dataset: Any, mapping: tuple[str, ...]) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    return [{"text": row["utterance"], "label": mapping[row["label"]]} for row in dataset], mapping


def main(argv: list[str] | None = None) -> int:
    return run_main(CONFIG, load_rows, argv)


if __name__ == "__main__":
    raise SystemExit(main())

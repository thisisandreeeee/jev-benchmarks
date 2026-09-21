"""Run BANKING77 test evaluations with SPACE-2 or TypeSafe Jev."""

from __future__ import annotations

from typing import Any

from benchmarks.space2_intents import ROOT, IntentConfig, main as run_main

CONFIG = IntentConfig(
    benchmark="banking77",
    dataset_id="PolyAI/banking77",
    dataset_revision="1fb62b1bb4635df59a8e1b2f2bc5e0643b2856c8",
    dataset_split="test",
    expected_rows=3_080,
    expected_labels=77,
    row_digest="e331bc3fa83d880409f6044836f709ae70de4193a4186f297385bbd6e5527808",
    instruction="Classify this banking customer request by choosing the most appropriate BANKING77 intent label.",
    space2_model="state_epoch_51",
    manifest=ROOT / "manifests" / "space2-banking77.json",
    author_prefix="banking-",
)


def load_rows(dataset: Any, _: tuple[str, ...]) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    labels = tuple(dataset.features["label"].names)
    return [{"text": row["text"], "label": labels[row["label"]]} for row in dataset], labels


def main(argv: list[str] | None = None) -> int:
    return run_main(CONFIG, load_rows, argv)


if __name__ == "__main__":
    raise SystemExit(main())

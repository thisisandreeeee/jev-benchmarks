"""Run CLINC150 in-scope test evaluations with SPACE-2 or TypeSafe Jev."""

from __future__ import annotations

from typing import Any

from benchmarks.space2_intents import ROOT, IntentConfig, main as run_main

CONFIG = IntentConfig(
    benchmark="clinc150",
    dataset_id="clinc/clinc_oos",
    dataset_config="plus",
    dataset_revision="155b9c710419136e17307b80d0a13e68cd46b4ec",
    dataset_split="test",
    expected_rows=4_500,
    expected_labels=150,
    row_digest="b5f47620c581ba0b5934b422f6e14a7cdd32ae8aa8020e78d4ee67ca85e0ce6b",
    instruction="Classify this request by choosing the most appropriate CLINC150 intent label.",
    space2_model="state_epoch_27",
    manifest=ROOT / "manifests" / "space2-clinc150.json",
    author_prefix="clinc-",
    dataset_details={"config": "plus", "filter": "intent != oos"},
)


def load_rows(dataset: Any, mapping: tuple[str, ...]) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    canonical = tuple(dataset.features["intent"].names)
    rows = [
        {"text": row["text"], "label": canonical[row["intent"]]}
        for row in dataset
        if canonical[row["intent"]] != "oos"
    ]
    return rows, mapping


def main(argv: list[str] | None = None) -> int:
    return run_main(CONFIG, load_rows, argv)


if __name__ == "__main__":
    raise SystemExit(main())

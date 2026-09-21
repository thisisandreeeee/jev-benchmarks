"""Pinned KEV-4B identity and local System One client."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from benchmarks.providers import SystemOneProvider

MODEL = "jaredpalmer/kev-4b"
MODEL_REVISION = "f780095d6511e6304868ff46ca2b74ee3eba29d8"
MODEL_CARD = f"https://huggingface.co/{MODEL}"
RUN = f"{MODEL}@{MODEL_REVISION}"
SLUG = "kev-4b"
API_MODEL = "kev-latest"
DEFAULT_BASE_URL = "http://127.0.0.1:8008"
REPOSITORY = "https://github.com/jaredpalmer/kev"
REPOSITORY_COMMIT = "e0bcf50153f1bda4ca6a8be5e12cbd5f9ebbce1c"
BASE_MODEL = "Qwen/Qwen3.5-4B-Base"
BASE_REVISION = "1001bb4d826a52d1f399e183466143f4da7b741b"

EXPOSURE = {
    "banking77": {
        "category": "benchmark_train_split",
        "details": "KEV decision-v7 includes BANKING77 training examples.",
    },
    "sst2": {
        "category": "related_task",
        "details": "KEV decision-v7 includes SST-5 and other sentiment datasets, but not SST-2.",
    },
    "clinc150": {"category": "no_known_task_specific_training"},
    "hwu64": {"category": "no_known_task_specific_training"},
    "stsb": {"category": "no_known_task_specific_training"},
}


def identity(benchmark: str) -> dict[str, Any]:
    return {
        "provider": "kev-local",
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "model_card": MODEL_CARD,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "repository": {"url": REPOSITORY, "commit": REPOSITORY_COMMIT},
        "training_exposure": EXPOSURE[benchmark],
        "temperature": 1.0,
    }


def _validate_server(payload: Any) -> dict[str, Any]:
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise ValueError("KEV server returned an invalid model listing")
    match = next((item for item in models if isinstance(item, dict) and item.get("id") == API_MODEL), None)
    actual = None if match is None else match.get("run")
    pinned_snapshot = isinstance(actual, str) and Path(actual).name == MODEL_REVISION
    if actual != RUN and not pinned_snapshot:
        raise ValueError(f"KEV server runs {actual!r}; expected {RUN!r}")
    if match.get("base") != BASE_MODEL:
        raise ValueError(f"KEV server base is {match.get('base')!r}; expected {BASE_MODEL!r}")
    return {"server_run": match["run"], "server_base": match["base"]}


def provider() -> SystemOneProvider:
    base_url = os.environ.get("KEV_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    with urlopen(f"{base_url}/v1/models", timeout=10) as response:  # noqa: S310 - explicit user-configured endpoint
        metadata = _validate_server(json.load(response))
    return SystemOneProvider(
        API_MODEL,
        provider_name="kev",
        api_key="local",
        base_url=base_url,
        timeout=120,
        metadata=metadata,
    )

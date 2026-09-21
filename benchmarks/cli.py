"""Shared provider metadata and result reporting for benchmark entry points."""

from __future__ import annotations

import json
from importlib.metadata import version
from typing import Any, Callable, NamedTuple

# Dependency groups are named after the runtime a provider family relies on.
DEPENDENCY_GROUPS: dict[str, tuple[str, ...]] = {
    "typesafe": ("datasets", "typesafe-sdk"),
    "transformers": ("datasets", "torch", "transformers"),
    "sentence-transformers": ("datasets", "sentence-transformers", "torch", "transformers"),
}


class ProviderBinding(NamedTuple):
    """Pinned output slug, dependency group, and concurrency limit for one provider."""

    slug: str
    dependency_group: str
    concurrency: int | None = None

    def dependencies(self) -> dict[str, str]:
        return {name: version(name) for name in DEPENDENCY_GROUPS[self.dependency_group]}


def finish(provider: Any, runner: Callable[[], dict[str, Any]], *, total: bool = False) -> int:
    """Run a benchmark, close the provider, and print its JSON summary."""
    try:
        result = runner()
    finally:
        close = getattr(provider, "close", None)
        if close is not None:
            close()
    summary: dict[str, Any] = {"status": result["status"], "evaluated": result["evaluated"]}
    if total:
        summary["total"] = result["total"]
    summary.update(result["metrics"])
    print(json.dumps(summary, indent=2))
    return 0

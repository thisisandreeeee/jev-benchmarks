"""Provider adapters behind one contract.

The contract (questions, results, and the ``Provider`` protocol) lives in
``benchmarks.providers.contract``. Each adapter module wraps one model runtime
and imports its SDK lazily, so importing this package stays lightweight.
"""

from __future__ import annotations

from benchmarks.providers.contract import (
    Choice,
    ChoiceResult,
    Inference,
    Noul,
    NoulResult,
    Provider,
    Question,
    Result,
    ScalarScoreResult,
    Score,
    ScoreResult,
    _normalize_distribution,
)
from benchmarks.providers.huggingface import TransformersNoulProvider, _label_indices
from benchmarks.providers.nli import NliZeroShotProvider, _default_device, _nli_choice, _nli_noul
from benchmarks.providers.sentence_transformers import CrossEncoderScoreProvider
from benchmarks.providers.space2 import Space2Provider, _load_space2_model, _space2_choice
from benchmarks.providers.typesafe import TypeSafeProvider

__all__ = [
    "Choice",
    "ChoiceResult",
    "CrossEncoderScoreProvider",
    "Inference",
    "NliZeroShotProvider",
    "Noul",
    "NoulResult",
    "Provider",
    "Question",
    "Result",
    "ScalarScoreResult",
    "Score",
    "ScoreResult",
    "Space2Provider",
    "TransformersNoulProvider",
    "TypeSafeProvider",
]

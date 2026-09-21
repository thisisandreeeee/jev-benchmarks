"""Provider-independent questions and inference results.

This module is the frozen vocabulary shared by every adapter. It must not
import an SDK or a model runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol


@dataclass(frozen=True)
class Choice:
    instructions: str
    criteria: tuple[str, ...]
    type: ClassVar[str] = "choice"

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.type, "instructions": self.instructions, "criteria": list(self.criteria)}


@dataclass(frozen=True)
class Score:
    instructions: str
    criteria: tuple[str, ...]
    type: ClassVar[str] = "score"

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.type, "instructions": self.instructions, "criteria": list(self.criteria)}


@dataclass(frozen=True)
class Noul:
    instructions: str
    type: ClassVar[str] = "noul"

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.type, "instructions": self.instructions}


Question = Choice | Score | Noul


@dataclass(frozen=True)
class ChoiceResult:
    choice: str
    probabilities: dict[str, float]
    type: ClassVar[str] = "choice"

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.type, "choice": self.choice, "probabilities": self.probabilities}


@dataclass(frozen=True)
class ScoreResult:
    score: float
    probabilities: dict[str, float]
    legend: dict[str, str]
    type: ClassVar[str] = "score"

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "score": self.score,
            "probabilities": self.probabilities,
            "legend": self.legend,
        }


@dataclass(frozen=True)
class ScalarScoreResult:
    score: float
    type: ClassVar[str] = "score"

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.type, "score": self.score}


@dataclass(frozen=True)
class NoulResult:
    noul: float
    type: ClassVar[str] = "noul"

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.type, "noul": self.noul}


Result = ChoiceResult | ScoreResult | ScalarScoreResult | NoulResult


@dataclass(frozen=True)
class Inference:
    result: Result
    metadata: dict[str, Any]


class Provider(Protocol):
    def infer(self, state: str, question: Question) -> Inference: ...


def _normalize_distribution(probabilities: dict[str, float], candidates: set[str]) -> dict[str, float]:
    if set(probabilities) != candidates:
        raise ValueError("provider probabilities do not match the submitted candidates")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
        raise ValueError("provider probabilities must be finite and between zero and one")
    total = sum(probabilities.values())
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=0.01 + 1e-9):
        raise ValueError("provider probabilities must sum to one")
    return probabilities if total == 1.0 else {key: value / total for key, value in probabilities.items()}

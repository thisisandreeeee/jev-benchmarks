"""Provider-independent questions and inference results."""

from __future__ import annotations

import math
import time
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
class NoulResult:
    noul: float
    type: ClassVar[str] = "noul"

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.type, "noul": self.noul}


Result = ChoiceResult | ScoreResult | NoulResult


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


class TypeSafeProvider:
    """Adapter for the first-party TypeSafe SDK."""

    def __init__(self, model: str = "jev-1.13.0") -> None:
        from typesafe_sdk import TypeSafeClient

        self.model = model
        self.client = TypeSafeClient(model=model)

    def close(self) -> None:
        self.client.close()

    def infer(self, state: str, question: Question) -> Inference:
        from typesafe_sdk import Choice as SDKChoice
        from typesafe_sdk import Noul as SDKNoul
        from typesafe_sdk import Score as SDKScore

        if isinstance(question, Choice):
            sdk_question = SDKChoice(instructions=question.instructions, criteria=dict.fromkeys(question.criteria))
        elif isinstance(question, Score):
            sdk_question = SDKScore(instructions=question.instructions, criteria=question.criteria)
        else:
            sdk_question = SDKNoul(instructions=question.instructions)

        started = time.perf_counter()
        response = self.client.system_one(state=state, questions={"answer": sdk_question})
        latency_ms = round((time.perf_counter() - started) * 1000)
        answer = response.answers.get("answer")
        if answer is None:
            raise ValueError("TypeSafe response did not contain the requested answer")

        confidence = getattr(answer, "confidence", None)
        metadata: dict[str, Any] = {
            "provider": "typesafe",
            "model": response.model,
            "latency_ms": latency_ms,
            "usage": {key: value for key, value in response.usage.model_dump().items() if value is not None},
        }
        if confidence is not None:
            metadata["confidence"] = confidence

        if isinstance(question, Choice) and answer.type == "choice":
            probabilities = _normalize_distribution(dict(answer.probabilities), set(question.criteria))
            if answer.choice not in question.criteria:
                raise ValueError("provider choice is not a submitted candidate")
            result: Result = ChoiceResult(answer.choice, probabilities)
        elif isinstance(question, Score) and answer.type == "score":
            candidates = {str(index) for index in range(len(question.criteria))}
            probabilities = _normalize_distribution(
                {str(key): value for key, value in answer.probabilities.items()},
                candidates,
            )
            legend = {str(key): str(value) for key, value in answer.legend.items()}
            if set(legend) != candidates or not math.isfinite(answer.score):
                raise ValueError("provider score does not match the submitted rubric")
            result = ScoreResult(answer.score, probabilities, legend)
        elif isinstance(question, Noul) and answer.type == "noul":
            if not math.isfinite(answer.noul) or not 0 <= answer.noul <= 1:
                raise ValueError("provider noul must be finite and between zero and one")
            result = NoulResult(answer.noul)
        else:
            raise ValueError(f"provider returned {answer.type!r} for a {question.type!r} question")

        return Inference(result, metadata)

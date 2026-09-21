"""First-party TypeSafe SDK adapter."""

from __future__ import annotations

import math
import time
from typing import Any

from benchmarks.providers.contract import (
    Choice,
    ChoiceResult,
    Inference,
    Noul,
    NoulResult,
    Question,
    Result,
    Score,
    ScoreResult,
    _normalize_distribution,
)


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
            if set(legend) != candidates or not math.isfinite(answer.score) or not 0 <= answer.score <= len(question.criteria) - 1:
                raise ValueError("provider score does not match the submitted rubric")
            result = ScoreResult(answer.score, probabilities, legend)
        elif isinstance(question, Noul) and answer.type == "noul":
            if not math.isfinite(answer.noul) or not 0 <= answer.noul <= 1:
                raise ValueError("provider noul must be finite and between zero and one")
            result = NoulResult(answer.noul)
        else:
            raise ValueError(f"provider returned {answer.type!r} for a {question.type!r} question")

        return Inference(result, metadata)

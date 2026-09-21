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


def __getattr__(name: str) -> Any:
    """Keep the historical imports while loading the optional SPACE-2 adapter lazily."""
    if name in {"Space2Provider", "_space2_choice"}:
        from benchmarks.space2_provider import Space2Provider, _space2_choice

        return {"Space2Provider": Space2Provider, "_space2_choice": _space2_choice}[name]
    raise AttributeError(name)


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


def _label_indices(label2id: dict[str, int | str]) -> dict[str, int]:
    return {label.casefold(): int(index) for label, index in label2id.items()}


class TransformersNoulProvider:
    """Local Hugging Face binary classifier exposed as a Noul probability."""

    def __init__(self, model: str, revision: str, positive_label: str) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_id = model
        self.revision = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, revision=revision)
        self.model.eval()
        labels = _label_indices(self.model.config.label2id)
        try:
            self.positive_index = labels[positive_label.casefold()]
        except KeyError as error:
            raise ValueError(f"model does not define positive label {positive_label!r}") from error

    def infer(self, state: str, question: Question) -> Inference:
        if not isinstance(question, Noul):
            raise ValueError("TransformersNoulProvider only accepts Noul questions")

        import torch

        inputs = self.tokenizer(state, return_tensors="pt", truncation=True)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0].float().cpu().tolist()
        maximum = max(logits)
        weights = [math.exp(logit - maximum) for logit in logits]
        probability = weights[self.positive_index] / sum(weights)
        return Inference(
            NoulResult(probability),
            {
                "provider": "huggingface",
                "model": self.model_id,
                "revision": self.revision,
                "confidence": max(probability, 1 - probability),
            },
        )


class CrossEncoderScoreProvider:
    """Local SentenceTransformers cross-encoder exposed as a scalar score."""

    def __init__(self, model: str, revision: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model_id = model
        self.revision = revision
        self.model = CrossEncoder(model, revision=revision)

    def infer_pair(self, sentence1: str, sentence2: str) -> Inference:
        score = float(self.model.predict([(sentence1, sentence2)], show_progress_bar=False)[0])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("cross-encoder score must be finite and between zero and one")
        return Inference(
            ScalarScoreResult(score),
            {"provider": "sentence-transformers", "model": self.model_id, "revision": self.revision},
        )


def _nli_choice(entailment_logits: list[float], labels: tuple[str, ...]) -> ChoiceResult:
    """Softmax a candidate entailment logit vector into a choice distribution."""
    if len(entailment_logits) != len(labels) or not labels:
        raise ValueError("NLI entailment logits do not match the candidate labels")
    if any(not math.isfinite(value) for value in entailment_logits):
        raise ValueError("NLI entailment logits must be finite")
    maximum = max(entailment_logits)
    weights = [math.exp(value - maximum) for value in entailment_logits]
    total = sum(weights)
    probabilities = {label: weight / total for label, weight in zip(labels, weights)}
    return ChoiceResult(max(probabilities, key=probabilities.get), probabilities)


def _nli_noul(entailment_logits: list[float]) -> NoulResult:
    """Convert positive/negative entailment logits into a positive probability."""
    if len(entailment_logits) != 2 or any(not math.isfinite(value) for value in entailment_logits):
        raise ValueError("Noul NLI scoring requires two finite entailment logits")
    maximum = max(entailment_logits)
    weights = [math.exp(value - maximum) for value in entailment_logits]
    return NoulResult(weights[0] / sum(weights))


class NliZeroShotProvider:
    """Local NLI cross-encoder exposed as a zero-shot Choice or Noul provider."""

    def __init__(
        self,
        model: str,
        revision: str,
        labels: tuple[str, ...],
        verbalization: dict[str, str],
        template: str,
        *,
        positive_label: str | None = None,
        max_length: int = 512,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_id = model
        self.revision = revision
        self.labels = tuple(labels)
        self.verbalization = dict(verbalization)
        self.template = template
        self.positive_label = positive_label
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device(device) if device is not None else _default_device()
        missing = [label for label in self.labels if label not in self.verbalization]
        if missing:
            raise ValueError(f"NLI provider has no verbalization for {missing}")
        if positive_label is not None and positive_label not in self.labels:
            raise ValueError("NLI positive label is not in the configured candidates")
        print(f"loading {model}@{revision[:12]} on {self.device} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, revision=revision)
        self.model.to(self.device).eval()
        self.entailment_index = self._entailment_index(self.model.config.id2label)
        self.prepared: dict[str, ChoiceResult] = {}

    @staticmethod
    def _entailment_index(id2label: dict[Any, Any]) -> int:
        matches = [int(index) for index, label in id2label.items() if str(label).casefold().startswith("entail")]
        if len(matches) != 1:
            raise ValueError("NLI checkpoint must define exactly one entailment label")
        return matches[0]

    def _hypotheses(self, labels: tuple[str, ...]) -> list[str]:
        return [self.template.format(self.verbalization[label]) for label in labels]

    def _entailment_logits(self, pairs: list[tuple[str, str]]) -> list[float]:
        import torch

        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                encoded = self.tokenizer(
                    [premise for premise, _ in batch],
                    [hypothesis for _, hypothesis in batch],
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                scores.extend(logits[:, self.entailment_index].float().cpu().tolist())
        return scores

    def _metadata(self, confidence: float) -> dict[str, Any]:
        return {
            "provider": "nli-zero-shot",
            "model": self.model_id,
            "revision": self.revision,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "confidence": confidence,
        }

    def infer(self, state: str, question: Question) -> Inference:
        if isinstance(question, Choice):
            if set(question.criteria) != set(self.labels) or len(question.criteria) != len(self.labels):
                raise ValueError("NLI provider only accepts its complete candidate label set")
            result = self.prepared.get(state)
            if result is None:
                result = _nli_choice(self._entailment_logits([(state, item) for item in self._hypotheses(self.labels)]), self.labels)
                self.prepared[state] = result
            return Inference(result, self._metadata(max(result.probabilities.values())))
        if isinstance(question, Noul):
            if self.positive_label is None:
                raise ValueError("NLI provider is not configured for Noul questions")
            ordered = (self.positive_label, *(label for label in self.labels if label != self.positive_label))
            result = _nli_noul(self._entailment_logits([(state, item) for item in self._hypotheses(ordered)]))
            return Inference(result, self._metadata(max(result.noul, 1 - result.noul)))
        raise ValueError("NLI provider only accepts Choice or Noul questions")


def _default_device() -> Any:
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

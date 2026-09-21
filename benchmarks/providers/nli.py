"""Local NLI cross-encoder adapter for zero-shot Choice and Noul questions."""

from __future__ import annotations

import contextlib
import math
from typing import Any

from benchmarks.providers.contract import (
    Choice,
    ChoiceResult,
    Inference,
    Noul,
    NoulResult,
    Question,
)

DTYPE_CHOICES = ("fp32", "fp16", "bf16")
DEFAULT_DTYPE = "fp32"


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


def _default_device() -> Any:
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_dtype(dtype: str) -> Any:
    """Return the autocast dtype for a name, or ``None`` for full precision."""
    import torch

    if dtype == "fp32":
        return None
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {dtype}")


def _autocast(device: Any, dtype: Any) -> Any:
    """Autocast context for a resolved device and dtype; a no-op for fp32."""
    import torch

    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=dtype)


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
        dtype: str = DEFAULT_DTYPE,
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
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else _default_device()
        if dtype == "fp16" and self.device.type == "cpu":
            raise ValueError("fp16 is not supported on CPU; use bf16 or fp32")
        self.autocast_dtype = _resolve_dtype(dtype)
        missing = [label for label in self.labels if label not in self.verbalization]
        if missing:
            raise ValueError(f"NLI provider has no verbalization for {missing}")
        if positive_label is not None and positive_label not in self.labels:
            raise ValueError("NLI positive label is not in the configured candidates")
        print(f"loading {model}@{revision[:12]} on {self.device} ({dtype}, batch {batch_size}) ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, revision=revision)
        self.model.to(self.device).eval()
        self.entailment_index = self._entailment_index(self.model.config.id2label)
        self.prepared: dict[str, ChoiceResult] = {}
        self.execution = {"device": str(self.device), "dtype": dtype, "batch_size": batch_size}

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
                with _autocast(self.device, self.autocast_dtype):
                    logits = self.model(**encoded).logits
                scores.extend(logits[:, self.entailment_index].float().cpu().tolist())
        return scores

    def _metadata(self, confidence: float) -> dict[str, Any]:
        return {
            "provider": "nli-zero-shot",
            "model": self.model_id,
            "revision": self.revision,
            "device": str(self.device),
            "dtype": self.dtype,
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

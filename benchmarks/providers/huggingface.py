"""Local Hugging Face sequence-classification adapter."""

from __future__ import annotations

import math
from typing import Any

from benchmarks.providers.contract import Inference, Noul, NoulResult, Question


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

"""Local SentenceTransformers cross-encoder adapter."""

from __future__ import annotations

import math

from benchmarks.providers.contract import Inference, ScalarScoreResult


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

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


def _space2_choice(logits: list[float], labels: tuple[str, ...]) -> ChoiceResult:
    if len(logits) != len(labels) or not logits or any(not math.isfinite(value) for value in logits):
        raise ValueError("SPACE-2 logits do not match the label mapping")
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    total = sum(weights)
    probabilities = {label: weight / total for label, weight in zip(labels, weights)}
    return ChoiceResult(max(probabilities, key=probabilities.get), probabilities)


def _load_space2_model(checkpoint: str):
    """Construct the released intent architecture without its legacy training stack."""
    import torch
    from torch import nn

    class Embedder(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding = nn.Embedding(30_522, 768)
            self.pos_embedding = nn.Embedding(512, 768)
            self.type_embedding = nn.Embedding(2, 768)
            self.turn_embedding = nn.Embedding(17, 768)

        def forward(self, token, position, token_type, turn):
            return (
                self.token_embedding(token)
                + self.pos_embedding(position)
                + self.type_embedding(token_type)
                + self.turn_embedding(turn)
            )

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_qkv = nn.Linear(768, 2_304)
            self.linear_out = nn.Linear(768, 768)

        def forward(self, value, mask):
            query, key, content = self.linear_qkv(value).chunk(3, dim=-1)
            query = query.reshape(value.shape[0], value.shape[1], 12, 64).transpose(1, 2)
            key = key.reshape(value.shape[0], value.shape[1], 12, 64).permute(0, 2, 3, 1)
            content = content.reshape(value.shape[0], value.shape[1], 12, 64).transpose(1, 2)
            scores = torch.matmul(query, key) * (64 ** -0.5)
            scores.masked_fill_(~mask[:, None, None, :], float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            attended = torch.matmul(weights, content).transpose(1, 2).reshape(value.shape)
            return self.linear_out(attended)

    class FeedForward(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_hidden = nn.Sequential(nn.Linear(768, 3_072), nn.GELU())
            self.linear_out = nn.Linear(3_072, 768)

        def forward(self, value):
            return self.linear_out(self.linear_hidden(value))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()
            self.attn_norm = nn.LayerNorm(768, eps=1e-12)
            self.ff = FeedForward()
            self.ff_norm = nn.LayerNorm(768, eps=1e-12)

        def forward(self, value, mask):
            value = self.attn_norm(self.attn(value, mask) + value)
            return self.ff_norm(self.ff(value) + value)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedder = Embedder()
            self.embed_layer_norm = nn.LayerNorm(768, eps=1e-12)
            self.layers = nn.ModuleList(Block() for _ in range(12))
            self.intent_classifier = nn.Linear(768, 77)

        def forward(self, token):
            position = torch.arange(token.shape[1]).unsqueeze(0).expand_as(token)
            value = self.embedder(token, position, torch.ones_like(token), torch.ones_like(token))
            cls = self.embedder.token_embedding.weight[101].reshape(1, 1, 768).expand(token.shape[0], -1, -1)
            value = self.embed_layer_norm(torch.cat((cls, value), dim=1))
            mask = torch.cat((torch.ones((token.shape[0], 1), dtype=torch.bool), token.ne(0)), dim=1)
            for layer in self.layers:
                value = layer(value, mask)
            return self.intent_classifier(value[:, 0])

    model = Model()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = {
        "mlm_bias",
        "mlm_transform.0.weight",
        "mlm_transform.0.bias",
        "mlm_transform.2.weight",
        "mlm_transform.2.bias",
    }
    if missing or set(unexpected) != allowed:
        raise ValueError(f"unexpected SPACE-2 checkpoint layout: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


class Space2Provider:
    """Released SPACE-2 BANKING77 checkpoint exposed as a Choice provider."""

    def __init__(self, checkpoint: str, vocab: str, labels: tuple[str, ...], checkpoint_sha256: str) -> None:
        from transformers import BertTokenizer

        self.labels = labels
        self.checkpoint_sha256 = checkpoint_sha256
        self.tokenizer = BertTokenizer(vocab_file=vocab, do_lower_case=True)
        self.model = _load_space2_model(checkpoint)
        self.prepared: dict[str, ChoiceResult] = {}

    def _tokens(self, state: str) -> list[int]:
        import regex

        pieces: list[str] = []
        for token in map(str.strip, regex.split(r"(\W+)", state.lower())):
            if token:
                pieces.extend(self.tokenizer.tokenize(token))
        return [13, *self.tokenizer.convert_tokens_to_ids(pieces)[-50:], 7][-256:]

    def prepare(self, states: list[str], batch_size: int = 128) -> None:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        with torch.inference_mode():
            for start in range(0, len(states), batch_size):
                batch = states[start : start + batch_size]
                tokens = pad_sequence(
                    [torch.tensor(self._tokens(state)) for state in batch], batch_first=True, padding_value=0
                )
                logits = self.model(tokens).float().tolist()
                self.prepared.update((state, _space2_choice(values, self.labels)) for state, values in zip(batch, logits))

    def infer(self, state: str, question: Question) -> Inference:
        if not isinstance(question, Choice) or set(question.criteria) != set(self.labels):
            raise ValueError("Space2Provider only accepts its complete Choice label mapping")

        import torch

        result = self.prepared.get(state)
        if result is None:
            with torch.inference_mode():
                logits = self.model(torch.tensor([self._tokens(state)])).squeeze(0).float().tolist()
            result = _space2_choice(logits, self.labels)
        return Inference(
            result,
            {
                "provider": "space-2",
                "model": "state_epoch_51",
                "checkpoint_sha256": self.checkpoint_sha256,
                "confidence": max(result.probabilities.values()),
            },
        )


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

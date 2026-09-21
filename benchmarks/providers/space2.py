"""SPACE-2 model reconstruction and provider adapter."""

from __future__ import annotations

import math

from benchmarks.providers.contract import Choice, ChoiceResult, Inference, Question


def _space2_choice(logits: list[float], labels: tuple[str, ...]) -> ChoiceResult:
    if len(logits) != len(labels) or not logits or any(not math.isfinite(value) for value in logits):
        raise ValueError("SPACE-2 logits do not match the label mapping")
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    total = sum(weights)
    probabilities = {label: weight / total for label, weight in zip(labels, weights)}
    return ChoiceResult(max(probabilities, key=probabilities.get), probabilities)


def _load_space2_model(checkpoint: str, num_labels: int):
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
            self.intent_classifier = nn.Linear(768, num_labels)

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
    """Released SPACE-2 intent checkpoint exposed as a Choice provider."""

    def __init__(self, checkpoint: str, vocab: str, labels: tuple[str, ...], checkpoint_sha256: str) -> None:
        from transformers import BertTokenizer

        self.labels = labels
        self.checkpoint_sha256 = checkpoint_sha256
        self.tokenizer = BertTokenizer(vocab_file=vocab, do_lower_case=True)
        self.model_name = checkpoint.rsplit("/", 1)[-1].removesuffix(".model")
        self.model = _load_space2_model(checkpoint, len(labels))
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
                "model": self.model_name,
                "checkpoint_sha256": self.checkpoint_sha256,
                "confidence": max(result.probabilities.values()),
            },
        )

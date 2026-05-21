from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import random
from pathlib import Path
from typing import Iterable

from .tokenizer import CharTokenizer


BOS = "<BOS>"
EOS = "<EOS>"


class NGramLanguageModel:
    """Character-level n-gram model with additive smoothing and backoff."""

    def __init__(
        self,
        order: int = 5,
        alpha: float = 0.05,
        tokenizer: CharTokenizer | None = None,
    ) -> None:
        if order < 1:
            raise ValueError("order must be at least 1")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.order = order
        self.alpha = alpha
        self.tokenizer = tokenizer or CharTokenizer()
        self.counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.context_totals: Counter[tuple[str, ...]] = Counter()
        self.vocab: list[str] = []

    def fit(self, texts: Iterable[str]) -> None:
        stories = [text.strip() for text in texts if text.strip()]
        joined = "\n".join(stories)
        self.tokenizer = CharTokenizer.train(joined)
        self.vocab = self.tokenizer.vocab + [EOS]

        for text in stories:
            chars = list(text)
            padded = [BOS] * (self.order - 1) + chars + [EOS]
            for idx in range(self.order - 1, len(padded)):
                token = padded[idx]
                max_context = self.order - 1
                for context_size in range(max_context + 1):
                    context = tuple(padded[idx - context_size : idx])
                    self.counts[context][token] += 1
                    self.context_totals[context] += 1

    def next_token_distribution(self, context: tuple[str, ...]) -> dict[str, float]:
        context = context[-(self.order - 1) :] if self.order > 1 else tuple()
        vocab_size = len(self.vocab)

        while True:
            counter = self.counts.get(context)
            if counter:
                total = self.context_totals[context] + self.alpha * vocab_size
                return {
                    token: (counter[token] + self.alpha) / total
                    for token in self.vocab
                }
            if not context:
                uniform = 1.0 / vocab_size
                return {token: uniform for token in self.vocab}
            context = context[1:]

    def token_log_probability(self, context: tuple[str, ...], token: str) -> float:
        distribution = self.next_token_distribution(context)
        return math.log(distribution.get(token, self.alpha / (self.alpha * len(self.vocab))))

    def negative_log_likelihood(self, text: str) -> float:
        if not text.strip():
            return 0.0
        padded = [BOS] * (self.order - 1) + list(text.strip()) + [EOS]
        losses = []
        for idx in range(self.order - 1, len(padded)):
            context = tuple(padded[idx - self.order + 1 : idx])
            token = padded[idx]
            losses.append(-self.token_log_probability(context, token))
        return sum(losses) / len(losses)

    def perplexity(self, text: str) -> float:
        return math.exp(self.negative_log_likelihood(text))

    def generate(
        self,
        prompt: str = "",
        max_new_chars: int = 500,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> str:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        rng = random.Random(seed)
        output = list(prompt)
        context = [BOS] * (self.order - 1)
        for ch in output[-(self.order - 1) :]:
            context.append(ch)
        context = context[-(self.order - 1) :]

        for _ in range(max_new_chars):
            distribution = self.next_token_distribution(tuple(context))
            if top_k is not None:
                if top_k < 1:
                    raise ValueError("top_k must be positive")
                distribution = dict(
                    sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:top_k]
                )
            tokens, weights = zip(*distribution.items())
            adjusted = [weight ** (1.0 / temperature) for weight in weights]
            total = sum(adjusted)
            probabilities = [weight / total for weight in adjusted]
            token = rng.choices(tokens, probabilities, k=1)[0]
            if token == EOS:
                break
            output.append(token)
            context.append(token)
            context = context[-(self.order - 1) :]
        return "".join(output)

    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "alpha": self.alpha,
            "tokenizer": self.tokenizer.to_dict(),
            "vocab": self.vocab,
            "counts": {
                "\u241f".join(context): dict(counter)
                for context, counter in self.counts.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NGramLanguageModel":
        model = cls(
            order=int(data["order"]),
            alpha=float(data["alpha"]),
            tokenizer=CharTokenizer.from_dict(data["tokenizer"]),
        )
        model.vocab = list(data["vocab"])
        for encoded_context, counter in data["counts"].items():
            context = tuple(encoded_context.split("\u241f")) if encoded_context else tuple()
            model.counts[context] = Counter(counter)
            model.context_totals[context] = sum(counter.values())
        return model

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "NGramLanguageModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

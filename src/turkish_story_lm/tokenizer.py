from __future__ import annotations


class CharTokenizer:
    """A transparent character tokenizer that keeps Turkish letters intact."""

    def __init__(self, vocab: list[str] | None = None) -> None:
        self.vocab = vocab or []
        self.stoi = {token: idx for idx, token in enumerate(self.vocab)}
        self.itos = {idx: token for token, idx in self.stoi.items()}

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        vocab = sorted(set(text))
        return cls(vocab)

    def encode(self, text: str) -> list[int]:
        missing = sorted(set(text) - set(self.stoi))
        if missing:
            shown = "".join(missing[:10])
            raise ValueError(f"Tokenizer vocabulary does not contain: {shown!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[idx] for idx in token_ids)

    def to_dict(self) -> dict[str, list[str]]:
        return {"vocab": self.vocab}

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> "CharTokenizer":
        return cls(list(data["vocab"]))

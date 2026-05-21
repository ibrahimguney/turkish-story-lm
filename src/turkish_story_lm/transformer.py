from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TransformerConfig:
    vocab_size: int
    block_size: int = 96
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, data: dict) -> "TransformerConfig":
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for the Transformer model. "
            "Install it with: python -m pip install -e .[transformer]"
        ) from exc
    return torch, nn, F


torch, nn, F = require_torch()


class TinyTransformerLM(nn.Module):
    """A small causal Transformer language model for character sequences."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4 * config.n_embd,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layer)
        self.norm = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        batch_size, time_steps = idx.shape
        if time_steps > self.config.block_size:
            raise ValueError("sequence length is larger than block_size")

        positions = torch.arange(time_steps, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)[None, :, :]
        mask = torch.triu(
            torch.ones(time_steps, time_steps, device=idx.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.blocks(x, mask=mask)
        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * time_steps, -1),
                targets.reshape(batch_size * time_steps),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = 8):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            values, indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
            probs = F.softmax(values, dim=-1)
            choice = torch.multinomial(probs, num_samples=1)
            next_token = indices.gather(-1, choice)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

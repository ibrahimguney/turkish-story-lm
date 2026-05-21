from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from turkish_story_lm.tokenizer import CharTokenizer
from turkish_story_lm.transformer import TinyTransformerLM, TransformerConfig, require_torch


torch, _, _ = require_torch()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Turkish Transformer LM.")
    parser.add_argument("--model", default="runs/transformer_tr.pt")
    parser.add_argument("--data", default="data/stories/tr_mini_stories.txt")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    config = TransformerConfig.from_dict(checkpoint["config"])
    model = TinyTransformerLM(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    text = Path(args.data).read_text(encoding="utf-8")
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    losses = []
    for start in range(0, len(encoded) - config.block_size - 1, config.block_size):
        chunk = encoded[start : start + config.block_size + 1]
        x = chunk[:-1][None, :]
        y = chunk[1:][None, :]
        _, loss = model(x, y)
        losses.append(loss.item())

    mean_nll = sum(losses) / len(losses)
    print(f"chunks={len(losses)}")
    print(f"mean_nll={mean_nll:.4f}")
    print(f"mean_perplexity={math.exp(mean_nll):.4f}")


if __name__ == "__main__":
    main()

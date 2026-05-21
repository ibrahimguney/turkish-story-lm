from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from turkish_story_lm.tokenizer import CharTokenizer
from turkish_story_lm.transformer import TinyTransformerLM, TransformerConfig, require_torch


torch, _, _ = require_torch()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with a Turkish Transformer LM.")
    parser.add_argument("--model", default="runs/transformer_tr.pt")
    parser.add_argument("--prompt", nargs="+", default=["Bir", "sabah"])
    parser.add_argument("--length", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.model, map_location="cpu")
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])
    config = TransformerConfig.from_dict(checkpoint["config"])
    model = TinyTransformerLM(config)
    model.load_state_dict(checkpoint["model_state"])

    prompt = " ".join(args.prompt)
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    generated = model.generate(
        idx,
        max_new_tokens=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()
    text = tokenizer.decode(generated)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"saved {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

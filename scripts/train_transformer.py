from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from turkish_story_lm.tokenizer import CharTokenizer
from turkish_story_lm.transformer import TinyTransformerLM, TransformerConfig, require_torch


torch, _, _ = require_torch()


def get_batch(data, block_size: int, batch_size: int, device: str):
    max_start = len(data) - block_size - 1
    starts = torch.randint(max_start, (batch_size,))
    x = torch.stack([data[start : start + block_size] for start in starts])
    y = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, block_size: int, batch_size: int, eval_iters: int, device: str):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny Turkish Transformer LM.")
    parser.add_argument("--data", default="data/stories/tr_mini_stories.txt")
    parser.add_argument("--out", default="runs/transformer_tr.pt")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=96)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = Path(args.data).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.train(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if len(encoded) <= args.block_size + 1:
        raise SystemExit("Dataset is too small for the requested block size.")

    config = TransformerConfig(
        vocab_size=len(tokenizer.vocab),
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
    model = TinyTransformerLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    started = time.time()
    for step in range(1, args.steps + 1):
        x, y = get_batch(encoded, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            eval_loss = estimate_loss(
                model, encoded, args.block_size, args.batch_size, args.eval_iters, device
            )
            print(f"step={step} train_loss={loss.item():.4f} eval_loss={eval_loss:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "model_state": model.state_dict(),
            "training": {
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seconds": round(time.time() - started, 2),
                "device": device,
            },
        },
        out,
    )
    metadata_path = out.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "model": str(out),
                "config": config.to_dict(),
                "training": {
                    "steps": args.steps,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "device": device,
                },
                "vocab_size": len(tokenizer.vocab),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {out}")
    print(f"saved {metadata_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from turkish_story_lm import NGramLanguageModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Turkish story sample.")
    parser.add_argument("--model", default="runs/ngram_tr.json")
    parser.add_argument("--prompt", nargs="+", default=["Bir", "sabah"])
    parser.add_argument("--length", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    model = NGramLanguageModel.load(args.model)
    text = model.generate(
        prompt=" ".join(args.prompt),
        max_new_chars=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"saved {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from turkish_story_lm import NGramLanguageModel


def split_stories(text: str) -> list[str]:
    return [part.strip() for part in text.split("\n\n") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Turkish character n-gram LM.")
    parser.add_argument("--data", default="data/stories/tr_mini_stories.txt")
    parser.add_argument("--out", default="runs/ngram_tr.json")
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.001)
    args = parser.parse_args()

    text = Path(args.data).read_text(encoding="utf-8")
    stories = split_stories(text)
    model = NGramLanguageModel(order=args.order, alpha=args.alpha)
    model.fit(stories)
    model.save(args.out)

    print(f"trained stories={len(stories)} order={args.order} vocab={len(model.vocab)}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

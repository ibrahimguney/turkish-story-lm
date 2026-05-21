from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from turkish_story_lm import NGramLanguageModel
from train_ngram import split_stories


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Turkish n-gram LM.")
    parser.add_argument("--model", default="runs/ngram_tr.json")
    parser.add_argument("--data", default="data/stories/tr_mini_stories.txt")
    args = parser.parse_args()

    model = NGramLanguageModel.load(args.model)
    stories = split_stories(Path(args.data).read_text(encoding="utf-8"))
    nll_values = [model.negative_log_likelihood(story) for story in stories]
    ppl_values = [model.perplexity(story) for story in stories]

    print(f"stories={len(stories)}")
    print(f"mean_nll={sum(nll_values) / len(nll_values):.4f}")
    print(f"mean_perplexity={sum(ppl_values) / len(ppl_values):.4f}")


if __name__ == "__main__":
    main()

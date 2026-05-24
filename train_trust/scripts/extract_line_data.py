"""
Extract L+ / L- per-line training data for the syntax-benign classifiers.

UntrustVul's syntax-benign ensemble (slide 8) is trained on:
    L+   "non-benign" lines:  vulnerable lines from the BigVul commits
    L-   "benign" lines:      non-vulnerable lines from the same commits

This script walks the BigVul training split and emits a CSV with two columns:

    line, label

where label = 0 for non-benign (L+) and label = 1 for benign (L-).

Then `train_syntax_classifier.py` trains a binary classifier on this CSV.

Usage:
    python -m train_trust.scripts.extract_line_data \\
        --config config.yaml \\
        --out data/line_dataset.csv
"""

from __future__ import annotations
import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from train_trust.data.bigvul import BigVulDataset


# Lines that are pure punctuation / scaffolding — discard from L- to avoid
# poisoning the dataset with trivial benign labels (matches UntrustVul protocol).
TRIVIAL_LINE_RE = re.compile(r"^\s*[{};]\s*$|^\s*$|^\s*//.*$|^\s*/\*.*\*/\s*$")


def is_trivial(line: str) -> bool:
    return bool(TRIVIAL_LINE_RE.match(line))


def extract_lines(ds, neg_per_pos: int = 3):
    """For each vulnerable function in `ds`:
        - emit every vulnerable line with label=0 (non-benign / L+)
        - emit up to `neg_per_pos` random non-vulnerable, non-trivial lines as label=1 (L-)
    """
    rng_state = 0  # deterministic-ish; uses Python's id-based ordering instead of random

    pos_rows, neg_rows = [], []

    for sample in ds.samples:
        if sample.label != 1 or not sample.vulnerable_lines:
            continue
        vul_set = set(sample.vulnerable_lines)
        for i, line in enumerate(sample.code_lines):
            if i in vul_set:
                if not is_trivial(line):
                    pos_rows.append({"line": line.strip(), "label": 0})
        # Sample negatives: non-vulnerable, non-trivial lines
        candidates = [
            line.strip() for i, line in enumerate(sample.code_lines)
            if i not in vul_set and not is_trivial(line) and line.strip()
        ]
        # Take up to neg_per_pos * |vul_lines| negatives
        n_take = min(len(candidates), neg_per_pos * len(vul_set))
        # Simple deterministic pick (every k-th)
        if candidates and n_take > 0:
            step = max(1, len(candidates) // n_take)
            for line in candidates[::step][:n_take]:
                neg_rows.append({"line": line, "label": 1})

    return pos_rows, neg_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="data/line_dataset.csv")
    ap.add_argument("--split", default="train",
                    help="which BigVul split to extract from")
    ap.add_argument("--neg_per_pos", type=int, default=3,
                    help="how many L- lines to sample per L+ line")
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if cfg["data"]["dataset"] != "bigvul":
        raise SystemExit("extract_line_data.py requires dataset=bigvul")

    ds = BigVulDataset(
        cfg["data"]["bigvul_path"],
        split=args.split,
        max_lines=cfg["model"]["max_lines"],
        max_samples=args.max_samples,
    )

    pos_rows, neg_rows = extract_lines(ds, neg_per_pos=args.neg_per_pos)
    rows = pos_rows + neg_rows
    df = pd.DataFrame(rows)
    df = df[df["line"].str.len() > 0]                 # drop empty after strip
    df = df.drop_duplicates(subset=["line", "label"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    counts = Counter(df["label"])
    print(f"\nWrote {len(df)} rows to {out_path}")
    print(f"  label=0 (L+, non-benign): {counts[0]}")
    print(f"  label=1 (L-, benign):     {counts[1]}")
    print(f"\nNext: train a classifier with")
    print(f"  python -m train_trust.scripts.train_syntax_classifier \\")
    print(f"      --data {out_path} --backbone microsoft/codebert-base \\")
    print(f"      --out checkpoints/syntax_codebert")


if __name__ == "__main__":
    main()

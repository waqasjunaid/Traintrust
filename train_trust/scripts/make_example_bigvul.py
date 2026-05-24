"""Generate a tiny BigVul-format CSV from the synthetic dataset.

Useful for verifying that the BigVul loader can parse files in the
expected format before pointing it at the real (multi-GB) BigVul.

Usage:
    python -m train_trust.scripts.make_example_bigvul --out data/bigvul_example
"""

from __future__ import annotations
import argparse
import random
from pathlib import Path

import pandas as pd

from train_trust.data.synthetic import _make_sample


def make_split(out_dir: Path, split: str, n: int, seed: int):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        s = _make_sample(rng)
        rows.append({
            "processed_func":   s.code,
            "target":           s.label,
            "flaw_line_index":  ",".join(str(v + 1) for v in s.vulnerable_lines),  # 1-indexed
        })
    df = pd.DataFrame(rows)
    out_path = out_dir / f"{split}.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(df)} rows, {df['target'].sum()} vuln)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bigvul_example")
    ap.add_argument("--n_train", type=int, default=64)
    ap.add_argument("--n_val", type=int, default=16)
    ap.add_argument("--n_test", type=int, default=16)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    make_split(out_dir, "train", args.n_train, seed=42)
    make_split(out_dir, "val",   args.n_val,   seed=43)
    make_split(out_dir, "test",  args.n_test,  seed=44)

    print(f"\nNow point config.yaml at this folder:")
    print(f"  data:")
    print(f"    dataset: bigvul")
    print(f"    bigvul_path: {out_dir}")


if __name__ == "__main__":
    main()

"""
Sanity-check your BigVul data load.

Prints a few sample functions, their detected vulnerable lines, and basic
statistics, so you can confirm the loader is parsing your specific
release correctly before launching a training run.

Usage:
    python -m train_trust.scripts.inspect_data --config config.yaml --split train
"""

from __future__ import annotations
import argparse

import yaml

from train_trust.data.bigvul import BigVulDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=3, help="how many samples to print")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if cfg["data"]["dataset"] != "bigvul":
        raise SystemExit("inspect_data.py only works with dataset=bigvul")

    ds = BigVulDataset(
        cfg["data"]["bigvul_path"],
        split=args.split,
        max_lines=cfg["model"]["max_lines"],
    )

    n_pos = sum(s.label for s in ds.samples)
    print(f"\n--- summary ---")
    print(f"total samples: {len(ds)}")
    print(f"vulnerable:    {n_pos} ({100 * n_pos / len(ds):.1f}%)")
    print(f"benign:        {len(ds) - n_pos}")

    n_with_lines = sum(1 for s in ds.samples if s.label == 1 and s.vulnerable_lines)
    print(f"vuln samples with line annotations: {n_with_lines} / {n_pos} "
          f"({100 * n_with_lines / max(n_pos, 1):.1f}%)")

    avg_len = sum(s.num_lines for s in ds.samples) / len(ds)
    print(f"avg lines per function: {avg_len:.1f}")

    print(f"\n--- first {args.n} samples ---")
    for i, s in enumerate(ds.samples[:args.n]):
        print(f"\n[{i}] label={s.label}  vuln_lines={s.vulnerable_lines}  num_lines={s.num_lines}")
        print("    " + "\n    ".join(s.code_lines[:8]))
        if s.num_lines > 8:
            print(f"    ... ({s.num_lines - 8} more)")


if __name__ == "__main__":
    main()

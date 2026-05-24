"""
Pre-warm the trust-signal cache (PDG + syntax-benign).

Run this ONCE before your first BigVul training run. It iterates over
every sample in train/val/test and computes:
    - benign_probs (from syntax-benign scorer)
    - reachability (from PDG / Joern stub)

Both arrays are stored in the disk cache. After this, training/eval read
from cache instead of recomputing — same results, much faster epochs.

Usage:
    python -m train_trust.scripts.preprocess --config config.yaml
    # or for a specific split:
    python -m train_trust.scripts.preprocess --config config.yaml --splits train,val
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import yaml
from tqdm import tqdm

from train_trust.data.bigvul import BigVulDataset
from train_trust.data.cache import TrustSignalCache
from train_trust.models.syntax_benign import StubBenignScorer
from train_trust.trust.pdg import compute_reachability_mask


def warm_split(split: str, cfg: dict):
    bigvul_path = cfg["data"]["bigvul_path"]
    cache_root = cfg["data"]["cache_root"]
    max_lines = cfg["model"]["max_lines"]

    ds = BigVulDataset(
        bigvul_path, split=split, max_lines=max_lines,
        max_samples=cfg["data"].get(f"{split}_size"),
    )
    cache = TrustSignalCache(cache_root, max_lines)
    scorer = StubBenignScorer()  # swap to real ensemble when ready

    t0 = time.time()
    for s in tqdm(ds.samples, desc=f"warm[{split}]"):
        if cache.get(s.code) is not None:
            continue
        bp = scorer(s.code_lines, max_lines)
        rc = compute_reachability_mask(s.code, max_lines)
        cache.put(s.code, bp, rc)

    dt = time.time() - t0
    stats = cache.stats
    print(
        f"[preprocess] split={split} done in {dt:.1f}s  "
        f"hits={stats['hits']} misses={stats['misses']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--splits", default="train,val,test",
                    help="comma-separated list of splits to warm")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if cfg["data"]["dataset"] != "bigvul":
        raise SystemExit(
            "preprocess.py only makes sense for dataset=bigvul. "
            f"Got dataset={cfg['data']['dataset']!r}."
        )
    if not cfg["data"].get("cache_root"):
        raise SystemExit(
            "Set data.cache_root in config.yaml before running preprocess.py."
        )

    Path(cfg["data"]["cache_root"]).mkdir(parents=True, exist_ok=True)

    for split in args.splits.split(","):
        split = split.strip()
        try:
            warm_split(split, cfg)
        except FileNotFoundError as e:
            print(f"[preprocess] skipping {split}: {e}")


if __name__ == "__main__":
    main()

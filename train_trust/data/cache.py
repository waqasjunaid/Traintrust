"""
Disk cache for preprocessed trust signals.

Computing PDG reachability and syntax-benign probabilities is expensive:
- PDG: spawns a Joern subprocess per function (~1-3 sec each)
- Syntax-benign: forward pass through three transformers per line

For a 150k-function BigVul split, that's hours per epoch if we recompute.
This cache stores both signals on first access and replays them after.

Cache key = SHA-256 of (code text, max_lines).  This means:
  - Same function in train/val/test splits gets cached once.
  - Changing max_lines invalidates correctly.
  - Editing a single character forces a re-extract (correctness-safe).

Layout on disk:
    <cache_root>/
        <hash[:2]>/<hash>.npz     # arrays: benign_probs, reachability
"""

from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class TrustSignalCache:
    """Two-array cache (benign_probs, reachability) keyed by code hash."""

    def __init__(self, cache_root: str | Path, max_lines: int):
        self.root = Path(cache_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_lines = max_lines
        self._hits = 0
        self._misses = 0

    def _key(self, code: str) -> str:
        h = hashlib.sha256()
        h.update(f"L={self.max_lines}\n".encode("utf-8"))
        h.update(code.encode("utf-8"))
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.npz"

    # ------------------------------------------------------------------

    def get(self, code: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return (benign_probs, reachability) if cached, else None."""
        path = self._path(self._key(code))
        if not path.exists():
            self._misses += 1
            return None
        try:
            data = np.load(path)
            self._hits += 1
            return data["benign_probs"].astype(np.float32), data["reachability"].astype(np.float32)
        except Exception:
            self._misses += 1
            return None

    def put(self, code: str, benign_probs: np.ndarray, reachability: np.ndarray) -> None:
        path = self._path(self._key(code))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            benign_probs=benign_probs.astype(np.float32),
            reachability=reachability.astype(np.float32),
        )

    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits":     self._hits,
            "misses":   self._misses,
            "hit_rate": self._hits / total if total else 0.0,
        }

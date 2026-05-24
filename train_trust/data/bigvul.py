"""
BigVul dataset loader.

BigVul (Fan et al., 2020) ships in several slightly different formats. The
two most common in the ML4SE literature:

    1. LineVul-cleaned CSV (awsm-research/LineVul):
       columns include `processed_func`, `target`, `flaw_line_index`
    2. UntrustVul-prepared CSV/JSON (Zenodo 15031367):
       columns include `func`, `label`, `vul_lines`

This loader auto-detects the schema by looking at column names, so you
shouldn't need to rewrite anything if you're switching between the two.

Expected directory layout:

    <bigvul_path>/
        train.csv      (or train.json)
        val.csv
        test.csv

If your release has a single CSV instead of pre-split files, run
`scripts/split_bigvul.py` (or your own splitter) first.

To use this loader, set in config.yaml:

    data:
      dataset: "bigvul"
      bigvul_path: "/path/to/your/bigvul/folder"
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from torch.utils.data import Dataset

from train_trust.data.synthetic import Sample


# -----------------------------------------------------------------------------
# Schema detection
# -----------------------------------------------------------------------------

# Map of (canonical field name) → (list of column names seen in the wild).
# Order matters: first match wins.
SCHEMA_ALIASES = {
    "code":     ["processed_func", "func", "func_before", "code", "function"],
    "label":    ["target", "label", "vul", "is_vulnerable"],
    "vul_lines":["flaw_line_index", "vul_lines", "flaw_lines",
                 "vuln_lines", "vulnerable_lines", "flaw_line"],
}


def _detect_columns(df: pd.DataFrame) -> dict:
    """Map canonical names to actual column names. Raises on missing essentials."""
    found = {}
    for canon, aliases in SCHEMA_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                found[canon] = a
                break

    missing = [c for c in ("code", "label") if c not in found]
    if missing:
        raise ValueError(
            f"BigVul CSV is missing required columns {missing}. "
            f"Saw columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}. "
            f"Add an alias to SCHEMA_ALIASES in train_trust/data/bigvul.py."
        )
    return found


# -----------------------------------------------------------------------------
# Vulnerable-line parsing
# -----------------------------------------------------------------------------

def parse_vul_lines(value, base_index: int = 1) -> List[int]:
    """Parse the `flaw_line_index` field into a list of 0-indexed line numbers.

    Accepts these representations:
        "3,7,12"          (comma-separated string, 1-indexed — most common)
        "[3, 7, 12]"      (JSON-like list, 1-indexed)
        [3, 7, 12]        (Python list)
        3                 (single int / numpy int / float-with-no-fraction)
        ""                (empty / NaN — returns [])

    Some releases use 1-indexed lines (LineVul, BigVul raw); a few use
    0-indexed. Set `base_index=0` if your data is already 0-indexed.
    """
    import math

    if value is None:
        return []
    # NaN check (works for float and pandas NA)
    try:
        if isinstance(value, float) and math.isnan(value):
            return []
    except (TypeError, ValueError):
        pass

    if isinstance(value, list):
        return [int(v) - base_index for v in value]

    # Numeric scalar (int, numpy int, or float that's actually integral).
    # We cover both Python int and any numpy/pandas numeric via duck typing.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not value.is_integer():
            # A non-integer float doesn't make sense as a line number; skip.
            return []
        return [int(value) - base_index]
    # Catch numpy.int64 etc. without importing numpy here
    if hasattr(value, "__int__") and not isinstance(value, str):
        try:
            return [int(value) - base_index]
        except (TypeError, ValueError):
            pass

    s = str(value).strip()
    if not s or s in {"[]", "nan", "NaN", "None", "<NA>"}:
        return []

    # Try JSON first (handles "[3, 7]")
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [int(v) - base_index for v in parsed]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fall back to splitting on common separators
    parts = re.split(r"[,;\s]+", s.strip("[](){}"))
    out = []
    for p in parts:
        p = p.strip()
        if p:
            try:
                out.append(int(float(p)) - base_index)
            except ValueError:
                continue
    return out


# -----------------------------------------------------------------------------
# Loader
# -----------------------------------------------------------------------------

class BigVulDataset(Dataset):
    """Loads a BigVul split (train / val / test) from disk and yields Samples.

    Args:
        bigvul_path: directory containing train.csv / val.csv / test.csv
                     (or .json variants).
        split:       "train" | "val" | "test".
        max_lines:   functions with more lines are truncated.
        max_samples: optional cap (useful for smoke tests on real data).
        base_index:  1 if your `flaw_line_index` is 1-indexed (BigVul default).
        skip_unparseable: if True, silently drop rows whose code can't be split
                     into a sensible number of lines.
    """

    def __init__(
        self,
        bigvul_path: str,
        split: str = "train",
        max_lines: int = 32,
        max_samples: Optional[int] = None,
        base_index: int = 1,
        skip_unparseable: bool = True,
    ):
        path = self._find_split_file(bigvul_path, split)
        df = self._load_dataframe(path)
        cols = _detect_columns(df)

        if max_samples:
            df = df.head(max_samples)

        self.samples: List[Sample] = []
        skipped = 0
        for _, row in df.iterrows():
            code = str(row[cols["code"]] or "").strip()
            if not code:
                skipped += 1
                continue

            lines = code.split("\n")
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                code = "\n".join(lines)

            label = int(row[cols["label"]])

            if "vul_lines" in cols and label == 1:
                raw = row[cols["vul_lines"]]
                vul_lines = [v for v in parse_vul_lines(raw, base_index)
                             if 0 <= v < len(lines)]
            else:
                vul_lines = []

            self.samples.append(Sample(
                code=code,
                code_lines=lines,
                label=label,
                vulnerable_lines=vul_lines,
                num_lines=len(lines),
            ))

        n_pos = sum(s.label for s in self.samples)
        print(
            f"[bigvul] split={split:5s}  loaded={len(self.samples):6d}  "
            f"vuln={n_pos}  benign={len(self.samples) - n_pos}  skipped={skipped}"
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

    # ------------------------------------------------------------------

    @staticmethod
    def _find_split_file(root: str, split: str) -> Path:
        """Look for <split>.csv, <split>.json, or <split>.jsonl under `root`."""
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(
                f"BigVul path {root} does not exist. Set data.bigvul_path in "
                "config.yaml to the directory containing your split files."
            )
        for suffix in (".csv", ".json", ".jsonl"):
            p = root / f"{split}{suffix}"
            if p.exists():
                return p
        # Fall back to <root>/<split>/data.csv etc.
        for suffix in (".csv", ".json", ".jsonl"):
            p = root / split / f"data{suffix}"
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Could not find a {split} file under {root}. "
            f"Expected one of: {split}.csv, {split}.json, {split}.jsonl, "
            f"or {split}/data.csv."
        )

    @staticmethod
    def _load_dataframe(path: Path) -> pd.DataFrame:
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".json":
            return pd.read_json(path)
        if path.suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        raise ValueError(f"Unsupported file type: {path.suffix}")

"""Unit tests for the BigVul loader and the trust-signal cache."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from train_trust.data.bigvul import BigVulDataset, parse_vul_lines, _detect_columns
from train_trust.data.cache import TrustSignalCache


# -----------------------------------------------------------------------------
# parse_vul_lines
# -----------------------------------------------------------------------------

def test_parse_vul_lines_comma_string():
    assert parse_vul_lines("3,7,12") == [2, 6, 11]   # 1→0 indexed


def test_parse_vul_lines_json_list():
    assert parse_vul_lines("[3, 7, 12]") == [2, 6, 11]


def test_parse_vul_lines_python_list():
    assert parse_vul_lines([3, 7]) == [2, 6]


def test_parse_vul_lines_empty():
    assert parse_vul_lines("") == []
    assert parse_vul_lines("[]") == []
    assert parse_vul_lines(None) == []
    assert parse_vul_lines(float("nan")) == []


def test_parse_vul_lines_numpy_int():
    """When pandas reads CSV with mixed integer/empty cells, values come back
    as numpy.int64 or numpy.float64 — parse_vul_lines must handle both."""
    import numpy as np
    assert parse_vul_lines(np.int64(3)) == [2]
    assert parse_vul_lines(np.float64(3.0)) == [2]
    assert parse_vul_lines(np.float64(float("nan"))) == []


def test_parse_vul_lines_zero_indexed():
    assert parse_vul_lines("3,7", base_index=0) == [3, 7]


# -----------------------------------------------------------------------------
# Schema detection
# -----------------------------------------------------------------------------

def test_detect_columns_linevul_format():
    df = pd.DataFrame(columns=["processed_func", "target", "flaw_line_index", "extra"])
    cols = _detect_columns(df)
    assert cols["code"] == "processed_func"
    assert cols["label"] == "target"
    assert cols["vul_lines"] == "flaw_line_index"


def test_detect_columns_alt_format():
    df = pd.DataFrame(columns=["func", "label", "vul_lines"])
    cols = _detect_columns(df)
    assert cols["code"] == "func"
    assert cols["label"] == "label"
    assert cols["vul_lines"] == "vul_lines"


def test_detect_columns_missing_required():
    df = pd.DataFrame(columns=["foo", "bar"])
    with pytest.raises(ValueError, match="missing required columns"):
        _detect_columns(df)


# -----------------------------------------------------------------------------
# Loader (with a temp CSV)
# -----------------------------------------------------------------------------

def _write_csv(path: Path, rows: list):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_bigvul_loader_smoke(tmp_path):
    train = tmp_path / "train.csv"
    _write_csv(train, [
        {"processed_func": "int f() {\n  strcpy(a, b);\n  return 0;\n}",
         "target": 1, "flaw_line_index": "2"},
        {"processed_func": "int g() {\n  return 1;\n}",
         "target": 0, "flaw_line_index": ""},
    ])
    ds = BigVulDataset(str(tmp_path), split="train", max_lines=16)
    assert len(ds) == 2
    assert ds[0].label == 1
    assert ds[0].vulnerable_lines == [1]    # 1-indexed input → 0-indexed output
    assert ds[1].label == 0
    assert ds[1].vulnerable_lines == []


def test_bigvul_loader_filters_oob_lines(tmp_path):
    """Lines beyond max_lines should be dropped from vulnerable_lines."""
    train = tmp_path / "train.csv"
    _write_csv(train, [
        {"processed_func": "\n".join(f"L{i}" for i in range(20)),
         "target": 1, "flaw_line_index": "5,12,18"},   # 1-indexed: lines 5/12/18
    ])
    ds = BigVulDataset(str(tmp_path), split="train", max_lines=10)
    # max_lines=10 keeps lines 0..9 (which were 1..10 in 1-indexed); line 5 (0-idx 4)
    # survives, lines 12 and 18 don't.
    assert ds[0].vulnerable_lines == [4]


def test_bigvul_loader_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        BigVulDataset(str(tmp_path), split="train")


# -----------------------------------------------------------------------------
# Cache
# -----------------------------------------------------------------------------

def test_cache_roundtrip(tmp_path):
    cache = TrustSignalCache(tmp_path / "cache", max_lines=8)
    code = "int main() { return 0; }"
    bp = np.array([0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    rc = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    assert cache.get(code) is None
    cache.put(code, bp, rc)
    out = cache.get(code)
    assert out is not None
    assert np.allclose(out[0], bp)
    assert np.allclose(out[1], rc)


def test_cache_invalidates_on_max_lines_change(tmp_path):
    """Same code with different max_lines should hash to different keys."""
    c8  = TrustSignalCache(tmp_path / "cache", max_lines=8)
    c16 = TrustSignalCache(tmp_path / "cache", max_lines=16)
    code = "int x;"
    arr = np.zeros(8, dtype=np.float32)
    c8.put(code, arr, arr)
    assert c16.get(code) is None     # different max_lines → different key
    assert c8.get(code) is not None

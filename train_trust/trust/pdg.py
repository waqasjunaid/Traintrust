"""
Program Dependence Graph (PDG) extraction — production version.

Public API:
    compute_reachability_mask(code: str, num_lines: int) -> np.ndarray

Returns a binary array of shape [num_lines] where entry i = 1 iff line i
"can reach a non-benign target through control/data dependencies".

Two implementations:
    * Real (Joern):   used when USE_JOERN=1 and joern-parse is on PATH.
    * Stub (regex):   pure Python; lets the project run end-to-end without
                      Joern installed.

Environment variables:
    USE_JOERN          "1" to enable the Joern path. Default "0".
    JOERN_TIMEOUT      Per-function timeout in seconds (default 60).
    JOERN_QUIET        "1" to suppress per-failure warnings (default "0").
"""

from __future__ import annotations
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def compute_reachability_mask(code: str, num_lines: int) -> np.ndarray:
    """Dispatch to Joern if available + enabled, else heuristic stub."""
    if _joern_enabled():
        try:
            return _joern_reachability(code, num_lines)
        except _JoernError as e:
            if os.environ.get("JOERN_QUIET", "0") != "1":
                log.warning("Joern failed (%s); falling back to stub for this sample.", e)
    return _stub_reachability(code, num_lines)


# -----------------------------------------------------------------------------
# Stub implementation
# -----------------------------------------------------------------------------

DANGEROUS_TOKENS = {
    "strcpy", "strcat", "sprintf", "gets", "memcpy", "memmove",
    "malloc", "alloca", "free", "system", "exec", "popen",
    "fopen", "open", "read", "write", "recv", "send",
    "printf", "scanf",
}


def _stub_reachability(code: str, num_lines: int) -> np.ndarray:
    """Heuristic: a line is 'reachable' if it contains a dangerous token,
    or if it is within 3 lines of one. Crude proxy for Joern's PDG.
    """
    lines = code.split("\n")[:num_lines]
    has_danger = np.zeros(num_lines, dtype=np.float32)
    for i, line in enumerate(lines):
        if any(tok in line for tok in DANGEROUS_TOKENS):
            has_danger[i] = 1.0
    mask = has_danger.copy()
    for i in range(num_lines):
        if has_danger[i] > 0:
            lo, hi = max(0, i - 3), min(num_lines, i + 4)
            mask[lo:hi] = 1.0
    return mask


# -----------------------------------------------------------------------------
# Joern implementation
# -----------------------------------------------------------------------------

class _JoernError(RuntimeError):
    """Internal: any Joern-related failure (parse, export, parse-DOT)."""


def _joern_enabled() -> bool:
    return (
        os.environ.get("USE_JOERN", "0") == "1"
        and shutil.which("joern-parse") is not None
        and shutil.which("joern-export") is not None
    )


def _joern_timeout() -> int:
    try:
        return int(os.environ.get("JOERN_TIMEOUT", "60"))
    except ValueError:
        return 60


def _joern_reachability(code: str, num_lines: int) -> np.ndarray:
    """Run Joern on a single function and extract PDG reachability.

    Process:
        1. Write code to a temp .c file
        2. joern-parse → CPG bin
        3. joern-export --repr pdg → DOT files
        4. Parse DOT, find dangerous-call target nodes
        5. Reverse-BFS to find lines that can reach those targets
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise _JoernError(f"networkx missing: {e}") from e

    timeout = _joern_timeout()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "fn.c"
        src.write_text(code)
        cpg = tmp / "cpg.bin"
        out = tmp / "pdg"

        try:
            subprocess.run(
                ["joern-parse", str(src), "--output", str(cpg)],
                check=True, capture_output=True, timeout=timeout,
            )
        except subprocess.CalledProcessError as e:
            raise _JoernError(f"joern-parse: {e.stderr.decode(errors='ignore')[:200]}") from e
        except subprocess.TimeoutExpired:
            raise _JoernError(f"joern-parse timeout ({timeout}s)")

        try:
            subprocess.run(
                ["joern-export", str(cpg), "--repr", "pdg", "--out", str(out)],
                check=True, capture_output=True, timeout=timeout,
            )
        except subprocess.CalledProcessError as e:
            raise _JoernError(f"joern-export: {e.stderr.decode(errors='ignore')[:200]}") from e
        except subprocess.TimeoutExpired:
            raise _JoernError(f"joern-export timeout ({timeout}s)")

        dot_files = sorted(out.glob("*.dot"))
        if not dot_files:
            raise _JoernError("no .dot files emitted")

        dot_path = max(dot_files, key=lambda p: p.stat().st_size)
        try:
            G = nx.drawing.nx_pydot.read_dot(str(dot_path))
        except Exception as e:
            raise _JoernError(f"failed to parse {dot_path.name}: {e}") from e

        # Identify dangerous call targets
        targets = []
        for n, attrs in G.nodes(data=True):
            label = (attrs.get("label", "") or "").lower()
            if any(tok in label for tok in DANGEROUS_TOKENS):
                targets.append(n)
        if not targets:
            return np.zeros(num_lines, dtype=np.float32)

        # Reverse-BFS to find ancestors
        rev = G.reverse(copy=False)
        reachable_nodes = set(targets)
        for t in targets:
            reachable_nodes.update(nx.descendants(rev, t))

        # Map nodes to source lines
        mask = np.zeros(num_lines, dtype=np.float32)
        for n in reachable_nodes:
            attrs = G.nodes[n]
            line_str = (
                attrs.get("LINE_NUMBER")
                or attrs.get("lineNumber")
                or attrs.get("LINE")
            )
            if line_str is None:
                continue
            try:
                line = int(str(line_str).strip('"')) - 1
                if 0 <= line < num_lines:
                    mask[line] = 1.0
            except (TypeError, ValueError):
                continue

        if mask.sum() == 0:
            raise _JoernError("no nodes mapped to source lines")
        return mask

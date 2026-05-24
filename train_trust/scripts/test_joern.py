"""
End-to-end test that Joern is installed and producing usable PDGs.

Usage:
    USE_JOERN=1 python -m train_trust.scripts.test_joern

Runs Joern on a tiny known-vulnerable C function and prints the resulting
reachability mask. Use this BEFORE preprocessing all of BigVul to confirm
your Joern setup is functional. If this fails, preprocess.py will fall
back silently to the stub for every sample (still produces a cache, just
not from real PDG analysis).
"""

from __future__ import annotations
import os
import sys

import numpy as np

from train_trust.trust.pdg import (
    compute_reachability_mask,
    _joern_enabled,
    _joern_reachability,
    _JoernError,
    _stub_reachability,
)


SAMPLE = """\
int copy_input(char *src) {
    char buf[64];
    int n = strlen(src);
    strcpy(buf, src);
    return n;
}
"""


def main():
    if not _joern_enabled():
        print(
            "Joern is NOT enabled. Either:\n"
            "  - USE_JOERN is not set to 1, or\n"
            "  - joern-parse / joern-export are not on PATH.\n"
            "\n"
            "Run:  export USE_JOERN=1\n"
            "And confirm:  which joern-parse joern-export"
        )
        sys.exit(1)

    print("Joern is enabled. Running on sample function...")
    print("---")
    print(SAMPLE)
    print("---")

    # Try Joern path explicitly so we get the real exception
    try:
        mask = _joern_reachability(SAMPLE, num_lines=8)
        print(f"Joern OK. Reachability mask: {mask.tolist()}")
        # Sanity check: line 4 has strcpy, so line 4 (idx 3) and earlier should be reachable
        if mask[3] == 1.0:
            print("✓ strcpy line correctly identified as reachable")
        else:
            print("! strcpy line not in reachable set — check Joern version")
    except _JoernError as e:
        print(f"Joern FAILED: {e}")
        print("\nFalling back to stub for comparison:")
        stub_mask = _stub_reachability(SAMPLE, num_lines=8)
        print(f"Stub mask: {stub_mask.tolist()}")
        sys.exit(2)


if __name__ == "__main__":
    main()

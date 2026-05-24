"""
Synthetic dataset for smoke-testing the pipeline.

Generates short C-like functions where:
    - Roughly half are "vulnerable" (contain a dangerous call like strcpy
      with an unchecked length).
    - Each vulnerable function has KNOWN vulnerable line indices, so we
      have ground-truth labels for L_attn-IoU and the IoU metric.

This is NOT a substitute for BigVul. It exists so the model, loss, and
training loop can be exercised on CPU in seconds without any downloads.
Run on this first to verify your environment works, then swap in the
real BigVul loader.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


VULN_TEMPLATES = [
    # Each template: (lines, indices of vulnerable lines)
    (
        [
            "int copy_data(char *src) {",
            "    char buf[64];",
            "    int len = strlen(src);",
            "    strcpy(buf, src);",   # vuln: no length check
            "    return strlen(buf);",
            "}",
        ],
        [3],
    ),
    (
        [
            "void format_msg(char *user) {",
            "    char out[128];",
            "    int n = strlen(user);",
            "    sprintf(out, \"hi %s\", user);",   # vuln: format-string
            "    log_message(out);",
            "}",
        ],
        [3],
    ),
    (
        [
            "char *load_file(int fd) {",
            "    int n = read(fd, NULL, 0);",
            "    char *buf = malloc(n);",            # vuln: n could be < 0
            "    read(fd, buf, n);",
            "    return buf;",
            "}",
        ],
        [2, 3],
    ),
]

BENIGN_TEMPLATES = [
    [
        "int add(int a, int b) {",
        "    int c = a + b;",
        "    return c;",
        "}",
    ],
    [
        "int max(int a, int b) {",
        "    if (a > b) return a;",
        "    return b;",
        "}",
    ],
    [
        "void print_hello(void) {",
        "    int x = 42;",
        "    return;",
        "}",
    ],
]


@dataclass
class Sample:
    code: str
    code_lines: List[str]
    label: int                       # 0 benign, 1 vulnerable
    vulnerable_lines: List[int]      # line indices known to be vulnerable
    num_lines: int


def _make_sample(rng: random.Random) -> Sample:
    if rng.random() < 0.5:
        lines, vuln_idx = rng.choice(VULN_TEMPLATES)
        label = 1
    else:
        lines = rng.choice(BENIGN_TEMPLATES)
        vuln_idx = []
        label = 0

    return Sample(
        code="\n".join(lines),
        code_lines=list(lines),
        label=label,
        vulnerable_lines=list(vuln_idx),
        num_lines=len(lines),
    )


class SyntheticVulnDataset(Dataset):
    """In-memory toy dataset.

    Used by tests/ and the smoke-test entrypoint.
    """

    def __init__(self, size: int, seed: int = 42):
        rng = random.Random(seed)
        self.samples: List[Sample] = [_make_sample(rng) for _ in range(size)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]

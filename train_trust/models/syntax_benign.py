"""
Syntax-benign classifier ensemble (slide 8, slide 11 step 2).

Three frozen classifiers (CodeBERT + GraphCodeBERT + UniXcoder) score each
code line as syntax-benign vs non-benign. The trio votes; outputs are
P(benign | line) ∈ [0, 1] per line.

This module provides:
    1. EnsembleClassifier — production: three frozen models with majority/average voting
    2. SingleModelScorer  — middle ground: one fine-tuned classifier
    3. StubBenignScorer   — heuristic for smoke tests

To produce trained classifiers from scratch, see:
    train_trust/scripts/extract_line_data.py     (build L+/L- training set)
    train_trust/scripts/train_syntax_classifier.py  (train one classifier)
"""

from __future__ import annotations
import re
from typing import List, Optional

import numpy as np


# -----------------------------------------------------------------------------
# 1) Stub for smoke tests
# -----------------------------------------------------------------------------

class StubBenignScorer:
    """Heuristic P_benign(line). For smoke tests only."""

    DANGEROUS_PATTERNS = [
        r"\bstrcpy\b", r"\bstrcat\b", r"\bsprintf\b", r"\bgets\b",
        r"\bmemcpy\b", r"\bmemmove\b", r"\bmalloc\b", r"\balloca\b",
        r"\bfree\b", r"\bsystem\b", r"\bexec\w*\b",
    ]
    OBVIOUSLY_BENIGN_PATTERNS = [
        r"^\s*$",
        r"^\s*//", r"^\s*/\*", r"^\s*\*",
        r"^\s*\}", r"^\s*\{",
        r"^\s*return\s*;",
        r"^\s*int\s+\w+\s*=\s*\d+\s*;$",
    ]

    def __call__(self, code_lines: List[str], num_lines: int) -> np.ndarray:
        out = np.full(num_lines, 0.5, dtype=np.float32)
        for i, line in enumerate(code_lines[:num_lines]):
            if any(re.search(p, line) for p in self.OBVIOUSLY_BENIGN_PATTERNS):
                out[i] = 0.95
            elif any(re.search(p, line) for p in self.DANGEROUS_PATTERNS):
                out[i] = 0.05
            else:
                out[i] = 0.6
        return out


# -----------------------------------------------------------------------------
# 2) Single-model scorer
# -----------------------------------------------------------------------------

class SingleModelScorer:
    """Use one fine-tuned classifier instead of three. Faster than the full
    ensemble; useful for ablations or when you only have one trained model.
    """

    def __init__(self, model_dir: str, device: str = "cpu", max_length: int = 64):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.max_length = max_length
        self._torch = torch

    def __call__(self, code_lines: List[str], num_lines: int) -> np.ndarray:
        out = np.full(num_lines, 0.5, dtype=np.float32)
        if not code_lines:
            return out
        lines = code_lines[:num_lines]
        enc = self.tok(
            lines, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        ).to(self.device)
        with self._torch.no_grad():
            logits = self.model(**enc).logits          # [N, 2]
            probs = self._torch.softmax(logits, dim=-1)[:, 1]
        out[:len(lines)] = probs.cpu().numpy().astype(np.float32)
        return out


# -----------------------------------------------------------------------------
# 3) Full ensemble
# -----------------------------------------------------------------------------

class EnsembleClassifier:
    """Three frozen classifiers vote on each line.

        mode='average':   mean of the three probabilities (smooth)
        mode='majority':  binarise each at 0.5 and majority-vote
    """

    def __init__(
        self,
        codebert_path: str,
        graphcodebert_path: str,
        unixcoder_path: str,
        device: str = "cpu",
        max_length: int = 64,
        mode: str = "average",
    ):
        if mode not in {"average", "majority"}:
            raise ValueError(f"mode must be 'average' or 'majority', got {mode!r}")
        self.mode = mode
        self.scorers = [
            SingleModelScorer(p, device=device, max_length=max_length)
            for p in [codebert_path, graphcodebert_path, unixcoder_path]
        ]

    def __call__(self, code_lines: List[str], num_lines: int) -> np.ndarray:
        probs = np.stack([s(code_lines, num_lines) for s in self.scorers], axis=0)
        # probs: [3, num_lines]
        if self.mode == "average":
            return probs.mean(axis=0).astype(np.float32)
        # majority vote
        votes = (probs > 0.5).astype(np.int32).sum(axis=0)
        majority_benign = (votes >= 2).astype(np.float32)
        return np.where(majority_benign > 0, 0.9, 0.1).astype(np.float32)


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def build_benign_scorer(cfg_block: Optional[dict]):
    """Build a scorer from a config block. See README for syntax."""
    if cfg_block is None or cfg_block.get("mode", "stub") == "stub":
        return StubBenignScorer()

    mode = cfg_block["mode"]
    device = cfg_block.get("device", "cpu")
    max_length = cfg_block.get("max_length", 64)

    if mode == "single":
        return SingleModelScorer(
            cfg_block["model_dir"], device=device, max_length=max_length,
        )
    if mode == "ensemble":
        return EnsembleClassifier(
            codebert_path     = cfg_block["codebert_path"],
            graphcodebert_path= cfg_block["graphcodebert_path"],
            unixcoder_path    = cfg_block["unixcoder_path"],
            device=device, max_length=max_length,
            mode=cfg_block.get("ensemble_mode", "average"),
        )
    raise ValueError(f"Unknown syntax_benign.mode: {mode!r}")

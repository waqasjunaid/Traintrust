"""
Collator: converts a list of `Sample`s into a model batch with all the
trust-signal tensors that L_trust needs.

Slide 11 step 1: PDGs are pre-computed.
Slide 11 step 2: syntax-benign probabilities come from the frozen ensemble.
Both happen here, NOT inside the training loop, so the loss can be a
straight forward pass over fixed targets.

For real datasets, set `cache_root` to enable on-disk caching of the
trust signals (otherwise they'd be recomputed every epoch — disastrous
on BigVul-scale data).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from train_trust.data.synthetic import Sample
from train_trust.data.cache import TrustSignalCache
from train_trust.models.syntax_benign import StubBenignScorer
from train_trust.trust.pdg import compute_reachability_mask


@dataclass
class Batch:
    input_ids: torch.Tensor          # [B, T]
    attention_mask: torch.Tensor     # [B, T]
    line_ids: torch.Tensor           # [B, T]   line index per token, -1 = pad
    labels: torch.Tensor             # [B]
    gt_lines: torch.Tensor           # [B, L]   binary mask of GT vulnerable lines
    benign_probs: torch.Tensor       # [B, L]
    reachability: torch.Tensor       # [B, L]


class TrustCollator:
    def __init__(
        self,
        tokenizer,
        max_seq_len: int,
        max_lines: int,
        cache_root: Optional[str] = None,
        benign_scorer = None,
    ):
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.max_lines = max_lines
        self.benign_scorer = benign_scorer or StubBenignScorer()
        self.cache = TrustSignalCache(cache_root, max_lines) if cache_root else None

    def __call__(self, samples: List[Sample]) -> Batch:
        B = len(samples)
        L = self.max_lines
        T = self.max_seq_len

        input_ids   = torch.zeros(B, T, dtype=torch.long)
        attn_mask   = torch.zeros(B, T, dtype=torch.long)
        line_ids    = torch.full((B, T), -1, dtype=torch.long)
        labels      = torch.zeros(B, dtype=torch.long)
        gt_lines    = torch.zeros(B, L, dtype=torch.float)
        benign_probs= torch.zeros(B, L, dtype=torch.float)
        reachable   = torch.zeros(B, L, dtype=torch.float)

        for i, s in enumerate(samples):
            ids, line_map = self._tokenize_with_line_map(s.code_lines)
            input_ids[i, :len(ids)] = torch.tensor(ids[:T])
            attn_mask[i, :len(ids)] = 1
            line_ids[i, :len(line_map)] = torch.tensor(line_map[:T])

            labels[i] = s.label
            for v in s.vulnerable_lines:
                if 0 <= v < L:
                    gt_lines[i, v] = 1.0

            bp, rc = self._get_or_compute_signals(s)
            benign_probs[i] = torch.tensor(bp)
            reachable[i]    = torch.tensor(rc)

        return Batch(
            input_ids=input_ids,
            attention_mask=attn_mask,
            line_ids=line_ids,
            labels=labels,
            gt_lines=gt_lines,
            benign_probs=benign_probs,
            reachability=reachable,
        )

    # -------------------------------------------------------------------------
    # Trust signals (cached)
    # -------------------------------------------------------------------------

    def _get_or_compute_signals(self, s: Sample):
        """Pull trust signals from cache or compute + store."""
        if self.cache is not None:
            cached = self.cache.get(s.code)
            if cached is not None:
                return cached
        bp = self.benign_scorer(s.code_lines, self.max_lines)
        rc = compute_reachability_mask(s.code, self.max_lines)
        if self.cache is not None:
            self.cache.put(s.code, bp, rc)
        return bp, rc

    # -------------------------------------------------------------------------
    # Tokenization helper
    # -------------------------------------------------------------------------

    def _tokenize_with_line_map(self, lines):
        """Tokenize line-by-line and remember which line each token came from."""
        ids = [self.tok.cls_token_id]
        line_map = [-1]                            # CLS belongs to no line

        for line_idx, line in enumerate(lines[:self.max_lines]):
            line_token_ids = self.tok.encode(line, add_special_tokens=False)
            ids.extend(line_token_ids)
            line_map.extend([line_idx] * len(line_token_ids))
            if len(ids) >= self.max_seq_len - 1:
                break

        ids.append(self.tok.sep_token_id)
        line_map.append(-1)

        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len - 1] + [self.tok.sep_token_id]
            line_map = line_map[:self.max_seq_len - 1] + [-1]

        return ids, line_map

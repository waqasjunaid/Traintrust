"""
Evaluation metrics from slide 11 of the deck.

Detection metrics (F1, AUC) come from sklearn — straightforward.
Trust metrics (IoU, T-score) need careful definition because they're
the heart of the empirical contribution.
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score


# -----------------------------------------------------------------------------
# Detection metrics
# -----------------------------------------------------------------------------

def detection_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Standard binary classification metrics on function-level vulnerability label."""
    probs = _softmax(logits)[:, 1]
    preds = (probs >= 0.5).astype(int)
    out = {
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "accuracy":  float((preds == labels).mean()),
    }
    if len(np.unique(labels)) > 1:
        out["auc"] = float(roc_auc_score(labels, probs))
    return out


# -----------------------------------------------------------------------------
# Trust metrics
# -----------------------------------------------------------------------------

def line_iou(pred_lines: np.ndarray, gt_lines: np.ndarray) -> float:
    """Hard IoU between two binary line masks (single sample)."""
    intersect = np.logical_and(pred_lines, gt_lines).sum()
    union = np.logical_or(pred_lines, gt_lines).sum()
    if union == 0:
        return 1.0  # both empty: vacuously perfect
    return float(intersect / union)


def average_iou(
    attentions: np.ndarray,    # [N, L]
    gt_lines:   np.ndarray,    # [N, L]   binary
    top_k: int = 5,
) -> float:
    """Average IoU between top-k attended lines and ground-truth vulnerable lines.

    This is the IoU operationalised in slides 11 and 16 (δ_IoU = 0.5).
    """
    ious = []
    for a, g in zip(attentions, gt_lines):
        if g.sum() == 0:
            continue  # skip benign samples
        k = min(top_k, len(a))
        topk_idx = np.argpartition(a, -k)[-k:]
        pred_mask = np.zeros_like(g)
        pred_mask[topk_idx] = 1
        ious.append(line_iou(pred_mask, g))
    return float(np.mean(ious)) if ious else 0.0


def trust_score(
    attentions: np.ndarray,         # [N, L]
    benign_probs: np.ndarray,       # [N, L]    from syntax-benign ensemble
    reachability: np.ndarray,       # [N, L]    PDG reachability mask
    top_k: int = 5,
) -> float:
    """T-score (slide 11, RQ5): proxy for how trustworthy each attention is.

    For each sample, look at the top-k attended lines. A line scores 1 if
    it is *either* (a) syntactically not benign  *or* (b) PDG-reachable
    to a non-benign line. The sample's T equals the fraction of its
    top-k attended lines that score 1; we average across samples.

    This mirrors UntrustVul's two-stage assessment and rewards models
    that attend to lines passing at least one of the two trust filters.
    """
    scores = []
    for a, b, r in zip(attentions, benign_probs, reachability):
        k = min(top_k, len(a))
        topk_idx = np.argpartition(a, -k)[-k:]
        # A line is "trustworthy" if non-benign OR reachable.
        non_benign = (b[topk_idx] < 0.5).astype(int)
        reaches    = (r[topk_idx] > 0.5).astype(int)
        ok = np.maximum(non_benign, reaches)
        scores.append(float(ok.mean()))
    return float(np.mean(scores)) if scores else 0.0


def trustworthiness_label(
    attention: np.ndarray,    # [L]   single sample
    gt_lines: np.ndarray,     # [L]   binary
    iou_threshold: float = 0.5,
    top_k: int = 5,
) -> int:
    """Slide 10: a prediction is *trustworthy* iff IoU(top-k, GT) > δ_IoU.

    Returns 1 (trustworthy) / 0 (untrustworthy).
    """
    if gt_lines.sum() == 0:
        return 1
    k = min(top_k, len(attention))
    topk_idx = np.argpartition(attention, -k)[-k:]
    pred_mask = np.zeros_like(gt_lines)
    pred_mask[topk_idx] = 1
    return int(line_iou(pred_mask, gt_lines) > iou_threshold)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()

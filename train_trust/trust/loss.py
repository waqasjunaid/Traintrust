"""
Trust-aware loss components.

Implements the three penalties described on slide 10 of the defense deck:

    L_trust  =  α_1 · L_attn-IoU  +  α_2 · L_syn  +  α_3 · L_pdg

and the full objective from slide 9:

    L_total  =  L_ce  +  λ · L_trust  +  λ_reg · L_reg

All components are differentiable in the model's *attention distribution* over
source-code lines. The non-differentiable parts (PDG reachability,
syntax-benign probabilities) are pre-computed and treated as fixed targets
during the backward pass — exactly the design hinted at on slide 11
(step 1: pre-compute PDGs; step 2: pre-train syntax-benign ensemble).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Individual penalty components
# -----------------------------------------------------------------------------

def attention_iou_loss(
    attention: torch.Tensor,        # [B, L]  line-level attention, sums to ~1 per row
    gt_lines: torch.Tensor,         # [B, L]  binary mask of ground-truth vulnerable lines
    eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable Dice-style soft IoU loss.

    Slide 10, α_1 component.  Pushes the model's attention distribution to
    overlap with the ground-truth vulnerable lines.

    Returns a scalar:  1 - Dice(attention, gt_lines).
    """
    # Dice = 2 * sum(a * g) / (sum(a) + sum(g))
    intersection = (attention * gt_lines).sum(dim=-1)
    denom = attention.sum(dim=-1) + gt_lines.sum(dim=-1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return (1.0 - dice).mean()


def syntax_benign_penalty(
    attention: torch.Tensor,        # [B, L]
    benign_probs: torch.Tensor,     # [B, L]  P(line is syntax-benign), from frozen ensemble
) -> torch.Tensor:
    """Slide 10, α_2 component.

    Penalises attention placed on lines the syntax-benign ensemble considers
    benign (high P_benign). The expected value of "attended benign mass"
    averaged over the batch.
    """
    return (attention * benign_probs).sum(dim=-1).mean()


def pdg_reachability_penalty(
    attention: torch.Tensor,        # [B, L]
    reachability: torch.Tensor,     # [B, L]  1 = line can reach a non-benign target in PDG, 0 = unreachable
) -> torch.Tensor:
    """Slide 10, α_3 component.

    Penalises attention placed on PDG-unreachable lines.
    `reachability` is computed offline by Joern (see train_trust/trust/pdg.py).
    """
    unreachable = 1.0 - reachability
    return (attention * unreachable).sum(dim=-1).mean()


# -----------------------------------------------------------------------------
# Combined trust loss
# -----------------------------------------------------------------------------

@dataclass
class TrustLossWeights:
    """Internal α weights and the top-level λ from config.yaml."""
    lambda_trust: float = 1.0
    lambda_reg: float = 0.0
    alpha_attn_iou: float = 1.0
    alpha_syn: float = 0.5
    alpha_pdg: float = 0.5


@dataclass
class TrustLossOutput:
    """Convenience container so we can log every component separately."""
    total: torch.Tensor
    ce: torch.Tensor
    trust: torch.Tensor
    attn_iou: torch.Tensor
    syn: torch.Tensor
    pdg: torch.Tensor

    def to_dict(self) -> dict:
        return {k: float(v.detach().cpu()) for k, v in self.__dict__.items()}


class TrustAwareLoss(torch.nn.Module):
    """The L_total objective from slide 9.

    Usage:

        criterion = TrustAwareLoss(weights)
        out = criterion(
            logits=logits,                      # [B, 2]
            labels=labels,                      # [B]   {0,1}
            attention=line_attention,           # [B, L]
            gt_lines=gt_vulnerable_lines,       # [B, L]
            benign_probs=benign_probs,          # [B, L]
            reachability=pdg_reachability,      # [B, L]
        )
        out.total.backward()
    """

    def __init__(self, weights: TrustLossWeights):
        super().__init__()
        self.w = weights
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention: torch.Tensor,
        gt_lines: Optional[torch.Tensor] = None,
        benign_probs: Optional[torch.Tensor] = None,
        reachability: Optional[torch.Tensor] = None,
        regularizer: Optional[torch.Tensor] = None,
    ) -> TrustLossOutput:
        ce_loss = self.ce(logits, labels)

        # If a sample has no GT lines (label = benign), zero out L_attn-IoU
        # for that sample so we don't push attention toward an empty target.
        # We do this with a soft mask over the batch.
        zero = torch.tensor(0.0, device=logits.device)

        if gt_lines is not None:
            # Only sum loss over samples that have at least one GT line.
            has_gt = (gt_lines.sum(dim=-1) > 0).float()
            if has_gt.sum() > 0:
                # Broadcast attention/gt to only the relevant samples.
                idx = has_gt.bool()
                a_sel = attention[idx]
                g_sel = gt_lines[idx]
                attn_iou = attention_iou_loss(a_sel, g_sel)
            else:
                attn_iou = zero
        else:
            attn_iou = zero

        syn = syntax_benign_penalty(attention, benign_probs) if benign_probs is not None else zero
        pdg = pdg_reachability_penalty(attention, reachability) if reachability is not None else zero

        trust = (
            self.w.alpha_attn_iou * attn_iou
            + self.w.alpha_syn      * syn
            + self.w.alpha_pdg      * pdg
        )

        reg = regularizer if regularizer is not None else zero
        total = ce_loss + self.w.lambda_trust * trust + self.w.lambda_reg * reg

        return TrustLossOutput(
            total=total,
            ce=ce_loss.detach(),
            trust=trust.detach(),
            attn_iou=attn_iou.detach() if isinstance(attn_iou, torch.Tensor) else zero,
            syn=syn.detach() if isinstance(syn, torch.Tensor) else zero,
            pdg=pdg.detach() if isinstance(pdg, torch.Tensor) else zero,
        )

"""Unit tests for the trust-aware loss.

These verify the mathematical properties the loss must have:
    1. Each component is differentiable and produces gradients on `attention`.
    2. L_attn-IoU is minimised (= 0) when attention exactly matches GT.
    3. L_syn is minimised when attention avoids high-benign lines.
    4. L_pdg is minimised when attention sits on reachable lines.
    5. Disabling a component via α=0 actually zeros it out.
"""

import torch
import pytest

from train_trust.trust.loss import (
    attention_iou_loss,
    syntax_benign_penalty,
    pdg_reachability_penalty,
    TrustAwareLoss,
    TrustLossWeights,
)


# -----------------------------------------------------------------------------
# Component-level tests
# -----------------------------------------------------------------------------

def test_attention_iou_perfect_match_is_zero():
    # attention concentrated entirely on the GT line → loss ≈ 0
    attn = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    gt   = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    loss = attention_iou_loss(attn, gt)
    assert loss.item() < 1e-3


def test_attention_iou_no_overlap_is_one():
    # attention everywhere except GT → loss approaches 1
    attn = torch.tensor([[0.5, 0.0, 0.5, 0.0]])
    gt   = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    loss = attention_iou_loss(attn, gt)
    assert loss.item() > 0.95


def test_attention_iou_is_differentiable():
    attn = torch.tensor([[0.25, 0.25, 0.25, 0.25]], requires_grad=True)
    gt   = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    loss = attention_iou_loss(attn, gt)
    loss.backward()
    assert attn.grad is not None
    # Gradient should push more mass onto index 1 (negative grad there).
    assert attn.grad[0, 1].item() < attn.grad[0, 0].item()


def test_syntax_benign_penalty_high_when_attending_benign():
    # Two scenarios: same attention, different benign maps.
    attn = torch.tensor([[0.0, 1.0, 0.0]])
    benign_high = torch.tensor([[0.0, 0.95, 0.0]])    # attended line IS benign
    benign_low  = torch.tensor([[0.95, 0.0, 0.95]])   # attended line is NOT benign
    high = syntax_benign_penalty(attn, benign_high)
    low  = syntax_benign_penalty(attn, benign_low)
    assert high.item() > low.item()


def test_pdg_penalty_zero_when_all_reachable():
    attn = torch.tensor([[0.5, 0.5, 0.0]])
    reach = torch.tensor([[1.0, 1.0, 1.0]])
    assert pdg_reachability_penalty(attn, reach).item() == pytest.approx(0.0)


def test_pdg_penalty_high_when_attending_unreachable():
    attn = torch.tensor([[1.0, 0.0]])
    reach = torch.tensor([[0.0, 1.0]])     # attention is on an unreachable line
    assert pdg_reachability_penalty(attn, reach).item() == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# Combined loss tests
# -----------------------------------------------------------------------------

def make_dummy_inputs(B=2, L=4, num_classes=2):
    logits = torch.randn(B, num_classes, requires_grad=True)
    labels = torch.tensor([1, 0])
    attn   = torch.softmax(torch.randn(B, L, requires_grad=True), dim=-1)
    attn   = attn.detach().requires_grad_(True)
    gt     = torch.zeros(B, L)
    gt[0, 1] = 1.0      # sample 0 has GT line at idx 1
    benign = torch.tensor([[0.1, 0.1, 0.9, 0.9],
                           [0.5, 0.5, 0.5, 0.5]])
    reach  = torch.tensor([[1.0, 1.0, 0.0, 0.0],
                           [1.0, 0.0, 1.0, 0.0]])
    return logits, labels, attn, gt, benign, reach


def test_total_loss_runs_and_has_gradient():
    weights = TrustLossWeights()
    crit = TrustAwareLoss(weights)
    logits, labels, attn, gt, benign, reach = make_dummy_inputs()
    out = crit(logits=logits, labels=labels, attention=attn,
               gt_lines=gt, benign_probs=benign, reachability=reach)
    out.total.backward()
    assert logits.grad is not None
    assert attn.grad is not None


def test_alpha_zero_disables_component():
    """Setting alpha_pdg=0 should make the L_pdg contribution vanish."""
    w_full = TrustLossWeights(alpha_pdg=1.0)
    w_off  = TrustLossWeights(alpha_pdg=0.0)
    crit_full = TrustAwareLoss(w_full)
    crit_off  = TrustAwareLoss(w_off)
    logits, labels, attn, gt, benign, reach = make_dummy_inputs()

    out_full = crit_full(logits=logits, labels=labels, attention=attn,
                         gt_lines=gt, benign_probs=benign, reachability=reach)
    out_off  = crit_off(logits=logits, labels=labels, attention=attn,
                        gt_lines=gt, benign_probs=benign, reachability=reach)

    # The two should differ exactly by the PDG term (with weight 1).
    diff = (out_full.total - out_off.total).item()
    assert abs(diff - out_full.pdg.item()) < 1e-5


def test_lambda_zero_recovers_pure_ce():
    """lambda_trust=0 ⇒ L_total = L_ce. Useful as a 'vanilla' control."""
    w = TrustLossWeights(lambda_trust=0.0)
    crit = TrustAwareLoss(w)
    logits, labels, attn, gt, benign, reach = make_dummy_inputs()
    out = crit(logits=logits, labels=labels, attention=attn,
               gt_lines=gt, benign_probs=benign, reachability=reach)
    assert abs(out.total.item() - out.ce.item()) < 1e-6


def test_handles_batch_with_no_gt_lines():
    """Some samples (benign ones) have no GT vulnerable lines; loss must not NaN."""
    crit = TrustAwareLoss(TrustLossWeights())
    logits = torch.randn(2, 2, requires_grad=True)
    labels = torch.tensor([0, 0])
    attn   = torch.softmax(torch.randn(2, 4), dim=-1).requires_grad_(True)
    gt     = torch.zeros(2, 4)               # all-zero: nothing is vulnerable
    benign = torch.full((2, 4), 0.5)
    reach  = torch.full((2, 4), 0.5)
    out = crit(logits=logits, labels=labels, attention=attn,
               gt_lines=gt, benign_probs=benign, reachability=reach)
    assert torch.isfinite(out.total)
    out.total.backward()

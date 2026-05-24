"""
Train-Trust training entry point.

Usage:
    python -m train_trust.train --config config.yaml

This script implements slide 11's training procedure:
    1. PDGs and syntax-benign signals are pre-computed by the collator.
    2. The detector is fine-tuned with L_total = L_ce + λ · L_trust.
    3. We log every loss component every `log_every` steps so you can watch
       L_trust decrease while L_ce stays flat (or improves).
"""

from __future__ import annotations
import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train_trust.data.synthetic import SyntheticVulnDataset
from train_trust.data.bigvul import BigVulDataset
from train_trust.data.collator import TrustCollator
from train_trust.models.detector import VulnDetector
from train_trust.models.local_detector import LocalVulnDetector, TinyTokenizer
from train_trust.models.syntax_benign import build_benign_scorer
from train_trust.trust.loss import TrustAwareLoss, TrustLossWeights
from train_trust.trust import metrics as M


# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train(cfg: dict):
    set_seed(cfg["train"]["seed"])
    device = pick_device(cfg["train"]["device"])
    print(f"[train] device = {device}")

    # ----- Data -----
    train_ds, val_ds = _build_datasets(cfg)

    tokenizer, model = _build_tokenizer_and_model(cfg)
    model = model.to(device)
    benign_scorer = build_benign_scorer(cfg.get("syntax_benign"))
    collator = TrustCollator(
        tokenizer,
        max_seq_len=cfg["model"]["max_seq_len"],
        max_lines=cfg["model"]["max_lines"],
        cache_root=cfg["data"].get("cache_root"),
        benign_scorer=benign_scorer,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"],
        shuffle=True, collate_fn=collator,
        num_workers=cfg["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"],
        shuffle=False, collate_fn=collator,
        num_workers=cfg["data"]["num_workers"],
    )

    # ----- Loss + optimizer -----
    weights = TrustLossWeights(
        lambda_trust=cfg["trust_loss"]["lambda_trust"],
        lambda_reg  =cfg["trust_loss"]["lambda_reg"],
        alpha_attn_iou=cfg["trust_loss"]["alpha_attn_iou"],
        alpha_syn   =cfg["trust_loss"]["alpha_syn"],
        alpha_pdg   =cfg["trust_loss"]["alpha_pdg"],
    )
    criterion = TrustAwareLoss(weights).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # ----- Train loop -----
    log_every = cfg["train"]["log_every"]
    step = 0
    best_f1 = -1.0
    best_tscore = -1.0
    save_dir = cfg["train"].get("save_dir")

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for batch in train_loader:
            batch = _move_batch(batch, device)
            logits, attention = model(
                batch.input_ids, batch.attention_mask, batch.line_ids
            )
            out = criterion(
                logits=logits, labels=batch.labels, attention=attention,
                gt_lines=batch.gt_lines,
                benign_probs=batch.benign_probs,
                reachability=batch.reachability,
            )
            optim.zero_grad()
            out.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if step % log_every == 0:
                d = out.to_dict()
                print(
                    f"[train] ep={epoch} step={step:4d}  "
                    f"L_total={d['total']:.4f}  L_ce={d['ce']:.4f}  "
                    f"L_trust={d['trust']:.4f} (iou={d['attn_iou']:.3f} "
                    f"syn={d['syn']:.3f} pdg={d['pdg']:.3f})"
                )
            step += 1

        # ----- Validation -----
        eval_metrics = evaluate(model, val_loader, device, cfg)
        print(
            f"[val]   ep={epoch}  F1={eval_metrics['f1']:.3f}  "
            f"avg_IoU={eval_metrics['avg_iou']:.3f}  T={eval_metrics['t_score']:.3f}"
        )

        if save_dir:
            ckpt_path = Path(save_dir)
            ckpt_path.mkdir(parents=True, exist_ok=True)
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": eval_metrics,
                "config": cfg,
            }

            # Save best model by F1 (primary checkpoint for paper results)
            if eval_metrics["f1"] > best_f1:
                best_f1 = eval_metrics["f1"]
                torch.save(ckpt_data, ckpt_path / "best_model.pt")
                print(f"  -> saved best-F1 checkpoint (F1={best_f1:.3f}) to {ckpt_path}")

            # Also save best model by T-score (secondary)
            if eval_metrics["t_score"] > best_tscore:
                best_tscore = eval_metrics["t_score"]
                torch.save(ckpt_data, ckpt_path / "best_tscore_model.pt")
                print(f"  -> saved best-T checkpoint (T={best_tscore:.3f}) to {ckpt_path}")

    return model, eval_metrics


@torch.no_grad()
def evaluate(model, loader, device, cfg) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    all_attn, all_gt, all_benign, all_reach = [], [], [], []

    for batch in loader:
        batch = _move_batch(batch, device)
        logits, attention = model(
            batch.input_ids, batch.attention_mask, batch.line_ids
        )
        all_logits.append(M.to_numpy(logits))
        all_labels.append(M.to_numpy(batch.labels))
        all_attn.append(M.to_numpy(attention))
        all_gt.append(M.to_numpy(batch.gt_lines))
        all_benign.append(M.to_numpy(batch.benign_probs))
        all_reach.append(M.to_numpy(batch.reachability))

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    attn   = np.concatenate(all_attn)
    gt     = np.concatenate(all_gt)
    benign = np.concatenate(all_benign)
    reach  = np.concatenate(all_reach)

    det = M.detection_metrics(logits, labels)
    avg_iou = M.average_iou(attn, gt, top_k=cfg["eval"]["attention_top_k"])
    tscore  = M.trust_score(attn, benign, reach, top_k=cfg["eval"]["attention_top_k"])
    return {**det, "avg_iou": avg_iou, "t_score": tscore}


def _move_batch(batch, device):
    """Move every tensor field of `Batch` to `device`."""
    for f in batch.__dataclass_fields__:
        v = getattr(batch, f)
        if isinstance(v, torch.Tensor):
            setattr(batch, f, v.to(device))
    return batch


def _build_datasets(cfg: dict):
    """Dispatch between synthetic toy data and real BigVul.

    Configure via:
        data:
          dataset: "synthetic" | "bigvul"
          bigvul_path: "<dir with train.csv / val.csv / test.csv>"
          train_size, val_size: caps (synthetic) or limits (bigvul)
          max_lines: same as model.max_lines
    """
    name = cfg["data"]["dataset"]
    seed = cfg["train"]["seed"]
    max_lines = cfg["model"]["max_lines"]

    if name == "synthetic":
        train_ds = SyntheticVulnDataset(cfg["data"]["train_size"], seed=seed)
        val_ds   = SyntheticVulnDataset(cfg["data"]["val_size"], seed=seed + 1)
        return train_ds, val_ds

    if name == "bigvul":
        path = cfg["data"]["bigvul_path"]
        train_ds = BigVulDataset(
            path, split="train", max_lines=max_lines,
            max_samples=cfg["data"].get("train_size"),
        )
        val_ds = BigVulDataset(
            path, split="val", max_lines=max_lines,
            max_samples=cfg["data"].get("val_size"),
        )
        return train_ds, val_ds

    raise ValueError(f"Unknown dataset: {name!r}. Use 'synthetic' or 'bigvul'.")


def _build_tokenizer_and_model(cfg: dict):
    """Dispatch between the local from-scratch model and a HuggingFace backbone.

    backbone: "local"     -> TinyTokenizer + LocalVulnDetector (no internet needed)
    backbone: "hf:<name>" -> AutoTokenizer + VulnDetector (downloads from HF Hub)
    """
    backbone = cfg["model"]["backbone"]
    num_lines = cfg["model"]["max_lines"]

    if backbone == "local":
        tokenizer = TinyTokenizer(vocab_size=4096)
        # Pre-fit the tokenizer's vocab on the synthetic data so encode is stable.
        # (Not strictly required — TinyTokenizer adds tokens lazily.)
        model = LocalVulnDetector(
            vocab_size=tokenizer.vocab_size,
            num_lines=num_lines,
            hidden=64, n_heads=4, n_layers=2,
            max_len=cfg["model"]["max_seq_len"],
        )
        return tokenizer, model

    if backbone.startswith("hf:"):
        hf_name = backbone[3:]
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model = VulnDetector(backbone=hf_name, num_lines=num_lines)
        return tokenizer, model

    raise ValueError(
        f"Unknown backbone: {backbone!r}. Use 'local' or 'hf:<model-name>'."
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()

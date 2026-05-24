"""
Evaluate a saved Train-Trust checkpoint on train/val/test split.

Usage:
    python -m train_trust.scripts.evaluate \
        --config configs/hf_bigvul.yaml \
        --checkpoint checkpoints/train_trust/best_model.pt \
        --split test
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train_trust.data.bigvul import BigVulDataset
from train_trust.data.collator import TrustCollator
from train_trust.models.detector import VulnDetector
from train_trust.models.syntax_benign import build_benign_scorer
from train_trust.trust import metrics as M


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


@torch.no_grad()
def evaluate_split(model, loader, device, cfg) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    all_attn, all_gt, all_benign, all_reach = [], [], [], []

    for batch in loader:
        for f in batch.__dataclass_fields__:
            v = getattr(batch, f)
            if isinstance(v, torch.Tensor):
                setattr(batch, f, v.to(device))

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

    # Additional: MCC (Matthews Correlation Coefficient)
    from sklearn.metrics import matthews_corrcoef
    probs = M._softmax(logits)[:, 1]
    preds = (probs >= 0.5).astype(int)
    mcc = float(matthews_corrcoef(labels, preds))

    return {**det, "mcc": mcc, "avg_iou": avg_iou, "t_score": tscore}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config YAML")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint dir or .pt file")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device(cfg["train"]["device"])
    print(f"[eval] device={device}  split={args.split}")

    # --- Load dataset ---
    max_lines = cfg["model"]["max_lines"]
    ds = BigVulDataset(
        cfg["data"]["bigvul_path"],
        split=args.split,
        max_lines=max_lines,
        max_samples=None,
    )
    print(f"[eval] loaded {len(ds)} samples from {args.split}")

    # --- Build model ---
    backbone = cfg["model"]["backbone"]
    hf_name = backbone[3:] if backbone.startswith("hf:") else backbone
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = VulnDetector(backbone=hf_name, num_lines=max_lines)

    # --- Load checkpoint ---
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "best_model.pt"
    print(f"[eval] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"[eval] checkpoint from epoch {ckpt.get('epoch', '?')}")
    if "metrics" in ckpt:
        print(f"[eval] checkpoint val metrics: {ckpt['metrics']}")

    # --- Build collator ---
    benign_scorer = build_benign_scorer(cfg.get("syntax_benign"))
    bs = args.batch_size or cfg["data"]["batch_size"]
    collator = TrustCollator(
        tokenizer,
        max_seq_len=cfg["model"]["max_seq_len"],
        max_lines=max_lines,
        cache_root=cfg["data"].get("cache_root"),
        benign_scorer=benign_scorer,
    )
    loader = DataLoader(
        ds, batch_size=bs, shuffle=False,
        collate_fn=collator, num_workers=0,
    )

    # --- Evaluate ---
    print(f"[eval] evaluating on {args.split} ({len(ds)} samples)...")
    results = evaluate_split(model, loader, device, cfg)

    # --- Print results ---
    print("\n" + "=" * 60)
    print(f"  RESULTS on {args.split.upper()} split")
    print("=" * 60)
    print(f"  F1:         {results['f1']:.4f}")
    print(f"  Precision:  {results['precision']:.4f}")
    print(f"  Recall:     {results['recall']:.4f}")
    print(f"  Accuracy:   {results['accuracy']:.4f}")
    if "auc" in results:
        print(f"  AUC-ROC:    {results['auc']:.4f}")
    print(f"  MCC:        {results['mcc']:.4f}")
    print(f"  avg_IoU:    {results['avg_iou']:.4f}")
    print(f"  T-score:    {results['t_score']:.4f}")
    print("=" * 60)

    # --- Save to JSON ---
    ckpt_id = Path(args.checkpoint).name if Path(args.checkpoint).is_dir() else Path(args.checkpoint).parent.name
    out_name = f"eval_{args.split}_{ckpt_id}.json"
    out_path = Path("results") / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] results saved to {out_path}")


if __name__ == "__main__":
    main()

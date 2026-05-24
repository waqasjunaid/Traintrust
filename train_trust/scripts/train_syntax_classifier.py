"""
Train a single syntax-benign classifier on L+/L- line data.

Usage:
    python -m train_trust.scripts.train_syntax_classifier \
        --data data/line_dataset.csv \
        --backbone microsoft/codebert-base \
        --out checkpoints/syntax_codebert \
        --epochs 3 --batch_size 32 --lr 2e-5

Repeat for graphcodebert-base and unixcoder-base to get the full ensemble.
Then point config.yaml at the three resulting directories.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm


class LineDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 64):
        self.lines = df["line"].tolist()
        self.labels = df["label"].tolist()
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        if not isinstance(text, str):
            text = str(text) if text is not None and text == text else ""
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_classifier(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[syntax] device={device}  backbone={args.backbone}")

    # Load data
    df = pd.read_csv(args.data)
    df["line"] = df["line"].fillna("").astype(str)
    df = df[df["line"].str.strip() != ""]
    print(f"[syntax] loaded {len(df)} lines  (label=0: {(df.label==0).sum()}, label=1: {(df.label==1).sum()})")

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    full_ds = LineDataset(df, tokenizer, max_length=args.max_length)

    # 90/10 train/val split
    n_val = max(1, int(0.1 * len(full_ds)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.backbone, num_labels=2
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    # Train
    best_f1 = 0.0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"ep{epoch}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            losses.append(loss.item())

        # Eval
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].cpu().tolist())

        f1 = f1_score(all_labels, all_preds, zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        avg_loss = np.mean(losses)
        print(f"[syntax] ep={epoch}  loss={avg_loss:.4f}  val_f1={f1:.3f}  val_acc={acc:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            print(f"  -> saved best model to {out_dir}")

    print(f"\n[syntax] done. Best val F1: {best_f1:.3f}")
    print(f"Model saved at: {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV from extract_line_data.py")
    ap.add_argument("--backbone", default="microsoft/codebert-base")
    ap.add_argument("--out", default="checkpoints/syntax_codebert")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=64)
    args = ap.parse_args()
    train_classifier(args)


if __name__ == "__main__":
    main()

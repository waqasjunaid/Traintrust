"""Split devign.csv into train/val/test sets."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/devign.csv")
    ap.add_argument("--output", default="data/devign_split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"[split] loaded {len(df)} samples (vuln={df.target.sum()}, benign={(df.target==0).sum()})")

    train_val, test = train_test_split(df, test_size=0.1, random_state=args.seed, stratify=df["target"])
    train, val = train_test_split(train_val, test_size=0.111, random_state=args.seed, stratify=train_val["target"])

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    print(f"[split] train: {len(train)} (vuln={train.target.sum()})")
    print(f"[split] val:   {len(val)} (vuln={val.target.sum()})")
    print(f"[split] test:  {len(test)} (vuln={test.target.sum()})")
    print(f"[split] saved to {out}/")

if __name__ == "__main__":
    main()

"""
Generate all experiment configs for CCF-A paper.

Usage:
    python -m train_trust.scripts.generate_configs --base configs/hf_bigvul.yaml

This creates:
  configs/ablation_iou_only.yaml
  configs/ablation_syn_only.yaml
  configs/ablation_pdg_only.yaml
  configs/ablation_iou_syn.yaml
  configs/ablation_iou_pdg.yaml
  configs/ablation_syn_pdg.yaml
  configs/lambda_0.1.yaml
  configs/lambda_0.5.yaml
  configs/lambda_2.0.yaml
  configs/lambda_5.0.yaml
"""

import argparse
import copy
from pathlib import Path
import yaml


ABLATIONS = {
    "ablation_iou_only": {"alpha_attn_iou": 1.0, "alpha_syn": 0.0, "alpha_pdg": 0.0},
    "ablation_syn_only": {"alpha_attn_iou": 0.0, "alpha_syn": 0.5, "alpha_pdg": 0.0},
    "ablation_pdg_only": {"alpha_attn_iou": 0.0, "alpha_syn": 0.0, "alpha_pdg": 0.5},
    "ablation_iou_syn":  {"alpha_attn_iou": 1.0, "alpha_syn": 0.5, "alpha_pdg": 0.0},
    "ablation_iou_pdg":  {"alpha_attn_iou": 1.0, "alpha_syn": 0.0, "alpha_pdg": 0.5},
    "ablation_syn_pdg":  {"alpha_attn_iou": 0.0, "alpha_syn": 0.5, "alpha_pdg": 0.5},
}

LAMBDAS = [0.1, 0.5, 2.0, 5.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base config (e.g. configs/hf_bigvul.yaml)")
    ap.add_argument("--epochs", type=int, default=5, help="Epochs per experiment")
    ap.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds (e.g. 42,123,456)")
    args = ap.parse_args()

    with open(args.base) as f:
        base = yaml.safe_load(f)

    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path("configs")
    out_dir.mkdir(exist_ok=True)

    count = 0

    # --- Ablation configs ---
    for name, alphas in ABLATIONS.items():
        for seed in seeds:
            cfg = copy.deepcopy(base)
            cfg["trust_loss"]["lambda_trust"] = 1.0
            cfg["trust_loss"]["alpha_attn_iou"] = alphas["alpha_attn_iou"]
            cfg["trust_loss"]["alpha_syn"] = alphas["alpha_syn"]
            cfg["trust_loss"]["alpha_pdg"] = alphas["alpha_pdg"]
            cfg["train"]["epochs"] = args.epochs
            cfg["train"]["seed"] = seed

            suffix = f"_s{seed}" if len(seeds) > 1 else ""
            save_name = f"{name}{suffix}"
            cfg["train"]["save_dir"] = f"checkpoints/{save_name}"

            fname = f"{save_name}.yaml"
            with open(out_dir / fname, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            print(f"  -> {out_dir / fname}")
            count += 1

    # --- Lambda sweep configs ---
    for lam in LAMBDAS:
        for seed in seeds:
            cfg = copy.deepcopy(base)
            cfg["trust_loss"]["lambda_trust"] = lam
            cfg["trust_loss"]["alpha_attn_iou"] = 1.0
            cfg["trust_loss"]["alpha_syn"] = 0.5
            cfg["trust_loss"]["alpha_pdg"] = 0.5
            cfg["train"]["epochs"] = args.epochs
            cfg["train"]["seed"] = seed

            suffix = f"_s{seed}" if len(seeds) > 1 else ""
            save_name = f"lambda_{lam}{suffix}"
            cfg["train"]["save_dir"] = f"checkpoints/{save_name}"

            fname = f"{save_name}.yaml"
            with open(out_dir / fname, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            print(f"  -> {out_dir / fname}")
            count += 1

    # --- Multi-seed configs for main experiments (Train-Trust + baseline) ---
    for seed in seeds:
        if seed == 42:
            continue  # already have seed=42 from original runs

        # Train-Trust with different seed
        cfg = copy.deepcopy(base)
        cfg["trust_loss"]["lambda_trust"] = 1.0
        cfg["train"]["epochs"] = args.epochs
        cfg["train"]["seed"] = seed
        cfg["train"]["save_dir"] = f"checkpoints/traintrust_s{seed}"
        fname = f"traintrust_s{seed}.yaml"
        with open(out_dir / fname, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"  -> {out_dir / fname}")
        count += 1

        # Baseline with different seed
        cfg = copy.deepcopy(base)
        cfg["trust_loss"]["lambda_trust"] = 0.0
        cfg["train"]["epochs"] = args.epochs
        cfg["train"]["seed"] = seed
        cfg["train"]["save_dir"] = f"checkpoints/baseline_s{seed}"
        fname = f"baseline_s{seed}.yaml"
        with open(out_dir / fname, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"  -> {out_dir / fname}")
        count += 1

    print(f"\nGenerated {count} config files.")
    print(f"\nTo run with 3 seeds for statistical significance:")
    print(f"  python -m train_trust.scripts.generate_configs --base {args.base} --seeds 42,123,456")


if __name__ == "__main__":
    main()

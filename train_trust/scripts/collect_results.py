"""
Collect and summarize all experiment results into formatted tables.

Usage:
    python -m train_trust.scripts.collect_results
"""

import json
import re
from pathlib import Path


def read_val_results_from_log(logfile: Path) -> list:
    results = []
    if not logfile.exists():
        return results
    with open(logfile) as f:
        for line in f:
            m = re.search(
                r'\[val\]\s+ep=(\d+)\s+F1=([\d.]+)\s+avg_IoU=([\d.]+)\s+T=([\d.]+)',
                line
            )
            if m:
                results.append({
                    "epoch": int(m.group(1)),
                    "f1": float(m.group(2)),
                    "avg_iou": float(m.group(3)),
                    "t_score": float(m.group(4)),
                })
    return results


def best_val_from_log(logfile: Path) -> dict:
    results = read_val_results_from_log(logfile)
    if not results:
        return {}
    return max(results, key=lambda x: x["f1"])


def find_test_json(name: str) -> dict:
    results_dir = Path("results")
    for pattern in [f"eval_test_{name}.json"]:
        p = results_dir / pattern
        if p.exists():
            with open(p) as f:
                return json.load(f)
    return {}


def find_val_log(name: str) -> dict:
    logs_dir = Path("logs")
    for logname in [f"{name}.txt", "step5_train.txt", "hf_bigvul.txt"]:
        r = best_val_from_log(logs_dir / logname)
        if r:
            return r
    return {}


def fmt(val, width=8, decimals=4):
    if val is None or val == '-':
        return f"{'—':>{width}}"
    return f"{float(val):>{width}.{decimals}f}"


def main():
    print("=" * 90)
    print("  TRAIN-TRUST: COMPLETE RESULTS SUMMARY")
    print("=" * 90)

    # TABLE 1
    print("\n" + "-" * 90)
    print("  TABLE 1: Train-Trust vs Baseline (RQ1)")
    print("-" * 90)
    print(f"  {'Method':<25} {'F1':>8} {'Prec':>8} {'Recall':>8} {'AUC':>8} {'MCC':>8} {'IoU':>8} {'T':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, label in [("train_trust", "Train-Trust (λ=1.0)"), ("baseline", "Baseline (λ=0.0)")]:
        r = find_test_json(name) or find_val_log(name)
        if r:
            print(f"  {label:<25} {fmt(r.get('f1'))} {fmt(r.get('precision'))} "
                  f"{fmt(r.get('recall'))} {fmt(r.get('auc'))} {fmt(r.get('mcc'))} "
                  f"{fmt(r.get('avg_iou'))} {fmt(r.get('t_score'))}")
        else:
            print(f"  {label:<25}  (no results found)")

    # TABLE 2 - Ablation
    print("\n" + "-" * 90)
    print("  TABLE 2: Ablation Study (RQ3)")
    print("-" * 90)
    print(f"  {'Components':<25} {'α_iou':>6} {'α_syn':>6} {'α_pdg':>6} {'F1':>8} {'Prec':>8} {'IoU':>8} {'T':>8}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    ablations = [
        ("baseline",            "Baseline (CE only)",   "—",   "—",   "—"),
        ("ablation_iou_only",   "L_attn-IoU only",      "1.0", "0.0", "0.0"),
        ("ablation_syn_only",   "L_syn only",           "0.0", "0.5", "0.0"),
        ("ablation_pdg_only",   "L_pdg only",           "0.0", "0.0", "0.5"),
        ("ablation_iou_syn",    "L_attn-IoU + L_syn",   "1.0", "0.5", "0.0"),
        ("ablation_iou_pdg",    "L_attn-IoU + L_pdg",   "1.0", "0.0", "0.5"),
        ("ablation_syn_pdg",    "L_syn + L_pdg",        "0.0", "0.5", "0.5"),
        ("train_trust",         "Full (all three)",      "1.0", "0.5", "0.5"),
    ]
    for name, label, a1, a2, a3 in ablations:
        r = find_test_json(name) or find_val_log(name)
        if r:
            print(f"  {label:<25} {a1:>6} {a2:>6} {a3:>6} "
                  f"{fmt(r.get('f1'))} {fmt(r.get('precision'))} "
                  f"{fmt(r.get('avg_iou'))} {fmt(r.get('t_score'))}")
        else:
            print(f"  {label:<25} {a1:>6} {a2:>6} {a3:>6}      (pending)")

    # TABLE 3 - Lambda
    print("\n" + "-" * 90)
    print("  TABLE 3: Lambda Sensitivity (RQ4)")
    print("-" * 90)
    print(f"  {'λ':<10} {'F1':>8} {'Prec':>8} {'Recall':>8} {'AUC':>8} {'MCC':>8} {'IoU':>8} {'T':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    lambdas = [
        ("0.0", "baseline"), ("0.1", "lambda_0.1"), ("0.5", "lambda_0.5"),
        ("1.0", "train_trust"), ("2.0", "lambda_2.0"), ("5.0", "lambda_5.0"),
    ]
    for lam_str, name in lambdas:
        r = find_test_json(name) or find_val_log(name)
        if r:
            print(f"  {lam_str:<10} {fmt(r.get('f1'))} {fmt(r.get('precision'))} "
                  f"{fmt(r.get('recall'))} {fmt(r.get('auc'))} {fmt(r.get('mcc'))} "
                  f"{fmt(r.get('avg_iou'))} {fmt(r.get('t_score'))}")
        else:
            print(f"  {lam_str:<10}      (pending)")

    # KEY FINDINGS
    print("\n" + "-" * 90)
    print("  KEY FINDINGS")
    print("-" * 90)
    best_lam, best_f1, best_r = None, 0, {}
    for lam_str, name in lambdas:
        r = find_test_json(name)
        if r and r.get("f1", 0) > best_f1:
            best_f1, best_lam, best_r = r["f1"], lam_str, r
    bl = find_test_json("baseline")
    if best_lam and bl:
        print(f"  Best λ: {best_lam}  (F1={best_f1:.4f})")
        print(f"  F1 improvement: +{(best_f1 - bl['f1'])*100:.2f}pp  ({bl['f1']:.4f} → {best_f1:.4f})")
        if bl.get('avg_iou', 0) > 0:
            pct = ((best_r.get('avg_iou',0)/bl['avg_iou'])-1)*100
            print(f"  IoU improvement: +{pct:.1f}%  ({bl['avg_iou']:.4f} → {best_r.get('avg_iou',0):.4f})")
    print("=" * 90)


if __name__ == "__main__":
    main()

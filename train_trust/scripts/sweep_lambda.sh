#!/usr/bin/env bash
# =============================================================================
# RQ4: Sensitivity sweep over λ (trust-loss weight)
#
# Usage:
#     bash train_trust/scripts/sweep_lambda.sh configs/bigvul_example.yaml
#
# Runs training for each λ value and logs results to sweep_results/.
# Each run uses a different config override for trust_loss.lambda_trust.
# =============================================================================

set -euo pipefail

CONFIG="${1:-configs/bigvul_example.yaml}"
OUT_DIR="sweep_results"
mkdir -p "$OUT_DIR"

LAMBDAS="0.0 0.1 0.25 0.5 1.0 2.0 4.0"

echo "========================================="
echo " RQ4 λ sweep — config: $CONFIG"
echo " λ values: $LAMBDAS"
echo "========================================="

for L in $LAMBDAS; do
    echo ""
    echo "--- λ = $L ---"

    # Create a temporary config with the overridden lambda
    TMP_CFG="$OUT_DIR/config_lambda_${L}.yaml"
    python3 -c "
import yaml, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['trust_loss']['lambda_trust'] = float($L)
with open('$TMP_CFG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
"

    # Run training and tee output to a log file
    LOG="$OUT_DIR/log_lambda_${L}.txt"
    python -m train_trust.train --config "$TMP_CFG" 2>&1 | tee "$LOG"

    echo "  -> log saved to $LOG"
done

echo ""
echo "========================================="
echo " Sweep complete. Logs in $OUT_DIR/"
echo " Parse the [val] lines to build the RQ4 table/chart."
echo "========================================="

# Quick summary: extract final val metrics from each log
echo ""
echo "λ       | F1      | avg_IoU | T-score"
echo "--------|---------|---------|--------"
for L in $LAMBDAS; do
    LOG="$OUT_DIR/log_lambda_${L}.txt"
    LAST_VAL=$(grep "\[val\]" "$LOG" | tail -1)
    F1=$(echo "$LAST_VAL"  | grep -oP 'F1=\K[0-9.]+')
    IOU=$(echo "$LAST_VAL" | grep -oP 'avg_IoU=\K[0-9.]+')
    T=$(echo "$LAST_VAL"   | grep -oP 'T=\K[0-9.]+')
    printf "%-7s | %-7s | %-7s | %s\n" "$L" "$F1" "$IOU" "$T"
done

#!/bin/bash
# =============================================================================
# Master experiment runner for Train-Trust CCF-A paper.
#
# Usage:
#   bash train_trust/scripts/run_all_experiments.sh
#
# This runs experiments in priority order:
#   Phase 1: Ablation study (6 experiments, ~15 hours)
#   Phase 2: Lambda sweep (4 experiments, ~10 hours)
#   Phase 3: Multi-seed runs (4 experiments, ~10 hours)
#   Phase 4: Test evaluation (all checkpoints, ~1 hour)
#
# Each experiment takes ~2.5 hours on RTX 3090.
# Total: ~36 hours if running sequentially.
#
# You can stop at any time and resume — completed experiments are skipped.
# =============================================================================

set -e
LOGDIR="logs"
RESULTDIR="results"
mkdir -p "$LOGDIR" "$RESULTDIR"

BASE_CONFIG="configs/hf_bigvul.yaml"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

run_experiment() {
    local config=$1
    local name=$2
    local logfile="$LOGDIR/${name}.txt"
    local checkpoint_dir
    checkpoint_dir=$(python -c "import yaml; c=yaml.safe_load(open('$config')); print(c['train']['save_dir'])")

    # Skip if checkpoint already exists
    if [ -f "${checkpoint_dir}/best_model.pt" ]; then
        echo -e "${GREEN}[SKIP]${NC} $name — checkpoint already exists at ${checkpoint_dir}"
        return 0
    fi

    echo -e "${YELLOW}[RUN]${NC} $name — config=$config  log=$logfile"
    echo "  Started: $(date)"

    python -m train_trust.train --config "$config" > "$logfile" 2>&1
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}[DONE]${NC} $name — $(grep '\[val\]' "$logfile" | tail -1)"
    else
        echo -e "${RED}[FAIL]${NC} $name — exit code $exit_code. Check $logfile"
    fi
    echo "  Finished: $(date)"
    echo ""
    return $exit_code
}

run_eval() {
    local config=$1
    local checkpoint=$2
    local split=$3
    local name=$4
    local logfile="$LOGDIR/eval_${name}_${split}.txt"
    local result_file="$RESULTDIR/eval_${split}_${name}.json"

    if [ -f "$result_file" ]; then
        echo -e "${GREEN}[SKIP]${NC} eval $name ($split) — result already exists"
        return 0
    fi

    if [ ! -f "${checkpoint}/best_model.pt" ]; then
        echo -e "${RED}[SKIP]${NC} eval $name ($split) — no checkpoint found"
        return 1
    fi

    echo -e "${YELLOW}[EVAL]${NC} $name on $split"
    python -m train_trust.scripts.evaluate \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --split "$split" > "$logfile" 2>&1

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[DONE]${NC} eval $name ($split)"
    else
        echo -e "${RED}[FAIL]${NC} eval $name ($split). Check $logfile"
    fi
}

echo "============================================================"
echo "  Train-Trust: CCF-A Experiment Suite"
echo "  Started: $(date)"
echo "============================================================"
echo ""

# ----- Phase 0: Generate configs if needed -----
echo "=== Phase 0: Generate experiment configs ==="
python -m train_trust.scripts.generate_configs --base "$BASE_CONFIG"
echo ""

# ----- Phase 1: Ablation Study (RQ3) -----
echo "=== Phase 1: Ablation Study (6 experiments) ==="
echo "  Testing which trust loss components contribute"
echo ""

run_experiment configs/ablation_iou_only.yaml "ablation_iou_only"
run_experiment configs/ablation_syn_only.yaml "ablation_syn_only"
run_experiment configs/ablation_pdg_only.yaml "ablation_pdg_only"
run_experiment configs/ablation_iou_syn.yaml  "ablation_iou_syn"
run_experiment configs/ablation_iou_pdg.yaml  "ablation_iou_pdg"
run_experiment configs/ablation_syn_pdg.yaml  "ablation_syn_pdg"

echo ""
echo "=== Phase 1 Complete ==="
echo ""

# ----- Phase 2: Lambda Sweep (RQ4) -----
echo "=== Phase 2: Lambda Sweep (4 experiments) ==="
echo "  Testing lambda = 0.1, 0.5, 2.0, 5.0"
echo "  (lambda=0.0 baseline and lambda=1.0 already done)"
echo ""

run_experiment configs/lambda_0.1.yaml "lambda_0.1"
run_experiment configs/lambda_0.5.yaml "lambda_0.5"
run_experiment configs/lambda_2.0.yaml "lambda_2.0"
run_experiment configs/lambda_5.0.yaml "lambda_5.0"

echo ""
echo "=== Phase 2 Complete ==="
echo ""

# ----- Phase 3: Multi-seed for Statistical Significance -----
echo "=== Phase 3: Multi-seed runs (optional, for significance tests) ==="
echo "  Run with --seeds 42,123,456 in generate_configs to enable"
echo ""

for cfg in configs/traintrust_s*.yaml configs/baseline_s*.yaml; do
    [ -f "$cfg" ] || continue
    name=$(basename "$cfg" .yaml)
    run_experiment "$cfg" "$name"
done

echo ""
echo "=== Phase 3 Complete ==="
echo ""

# ----- Phase 4: Evaluate ALL checkpoints on test set -----
echo "=== Phase 4: Test Set Evaluation ==="
echo ""

# Main experiments
run_eval "$BASE_CONFIG" "checkpoints/train_trust" "test" "traintrust"
run_eval "$BASE_CONFIG" "checkpoints/baseline" "test" "baseline"

# Ablation checkpoints
for abl in iou_only syn_only pdg_only iou_syn iou_pdg syn_pdg; do
    cfg="configs/ablation_${abl}.yaml"
    [ -f "$cfg" ] && run_eval "$cfg" "checkpoints/ablation_${abl}" "test" "ablation_${abl}"
done

# Lambda sweep checkpoints
for lam in 0.1 0.5 2.0 5.0; do
    cfg="configs/lambda_${lam}.yaml"
    [ -f "$cfg" ] && run_eval "$cfg" "checkpoints/lambda_${lam}" "test" "lambda_${lam}"
done

# Multi-seed checkpoints
for ckpt_dir in checkpoints/traintrust_s* checkpoints/baseline_s*; do
    [ -d "$ckpt_dir" ] || continue
    name=$(basename "$ckpt_dir")
    run_eval "$BASE_CONFIG" "$ckpt_dir" "test" "$name"
done

echo ""
echo "=== Phase 4 Complete ==="
echo ""

# ----- Phase 5: Collect and summarize results -----
echo "=== Phase 5: Results Summary ==="
python -m train_trust.scripts.collect_results
echo ""

echo "============================================================"
echo "  All experiments complete!"
echo "  Finished: $(date)"
echo "  Results saved to: $RESULTDIR/"
echo "============================================================"

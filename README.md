# Train-Trust

**Trustworthiness-Guided Loss Optimization for AI-Driven Vulnerability Detection**

A reference implementation of the loss-based trustworthiness approach. The
core idea (slide 9 of the defense deck):

```
L_total  =  L_ce  +  λ · L_trust  +  λ_reg · L_reg
L_trust  =  α₁ · L_attn-IoU  +  α₂ · L_syn  +  α₃ · L_pdg
```

Instead of filtering untrustworthy predictions *after* training (UntrustVul,
CausalVul), Train-Trust bakes trustworthiness directly into the training
objective. Trust signals come from three sources: ground-truth vulnerable
lines (when available), a frozen syntax-benign classifier ensemble, and a
Joern-extracted Program Dependence Graph.

---

## Project structure

```
train-trust/
├── config.yaml                          # all hyperparameters (λ, α weights, etc.)
├── requirements.txt
├── train_trust/
│   ├── data/
│   │   ├── synthetic.py                 # toy dataset for smoke tests
│   │   ├── bigvul.py                    # TODO: real BigVul loader
│   │   └── collator.py                  # builds batches with pre-computed trust signals
│   ├── models/
│   │   ├── detector.py                  # LineVul-style detector with line-level attention
│   │   └── syntax_benign.py             # frozen ensemble (real) + heuristic stub (smoke)
│   ├── trust/
│   │   ├── loss.py                      # ★ THE NOVEL CONTRIBUTION ★ — three penalty components
│   │   ├── pdg.py                       # Joern wrapper + heuristic fallback
│   │   └── metrics.py                   # F1, AUC, avg IoU, T-score
│   └── train.py                         # main training entry point
└── tests/
    └── test_loss.py                     # unit tests for the loss
```

---

## Quickstart (smoke test, ~2 minutes on CPU)

```bash
# 1. Set up a virtualenv
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run unit tests — these verify the loss math is correct
pytest tests/ -v

# 4. Run the smoke test on synthetic data
#    Uses a tiny BERT model so it runs in seconds on CPU.
python -m train_trust.train --config config.yaml
```

You should see something like:

```
[train] device = cpu
[train] ep=0 step=   0  L_total=0.94  L_ce=0.71  L_trust=0.23 (iou=0.40 syn=0.30 pdg=0.10)
[train] ep=0 step=   5  L_total=0.81  L_ce=0.69  L_trust=0.12 (iou=0.20 syn=0.20 pdg=0.04)
...
[val]   ep=2  F1=1.000  avg_IoU=0.300  T=0.620
```

`L_trust` should decrease over training while `L_ce` stays low — that's the
loss doing what the slides describe.

---

## Going from smoke test → real experiments

The smoke test uses three deliberate stand-ins so the code runs anywhere:

| Component | Smoke-test stand-in | Real implementation |
|---|---|---|
| Dataset | `SyntheticVulnDataset` (in-memory toy C functions) | **`BigVulDataset` — implemented and ready** (auto-detects schema, supports CSV/JSON) |
| Detector backbone | `prajjwal1/bert-tiny` (4-layer toy model) | `microsoft/graphcodebert-base` or `microsoft/codebert-base` — change `model.backbone` in config.yaml |
| Syntax-benign scorer | `StubBenignScorer` (regex heuristic) | `EnsembleClassifier` — train three classifiers on UntrustVul's L+/L− data, then implement |
| PDG | `_stub_reachability` (token-distance heuristic) | Joern — set `USE_JOERN=1` and install joern-cli |

You can swap them in one at a time. After each swap, re-run the smoke test;
if it still trains, your integration is correct.

### Wiring up real BigVul data

```bash
# 1. (Optional) Verify the loader on a tiny example in BigVul format
python -m train_trust.scripts.make_example_bigvul --out data/bigvul_example
python -m train_trust.scripts.inspect_data --config configs/bigvul_example.yaml

# 2. Point config at YOUR BigVul folder
#    Expected: <bigvul_path>/{train,val,test}.csv (or .json)
#    Common columns the loader recognises automatically:
#       processed_func / func / func_before / code     — the function source
#       target / label / vul                            — 0/1 vulnerability label
#       flaw_line_index / vul_lines / flaw_lines        — 1-indexed vulnerable lines

# 3. Pre-warm the trust-signal cache (PDG + syntax-benign).
#    Run this ONCE before training; subsequent epochs read from cache.
python -m train_trust.scripts.preprocess --config config.yaml

# 4. Train
python -m train_trust.train --config config.yaml
```

If the loader complains about missing columns, your release uses a column
name not in the alias list. Add it to `SCHEMA_ALIASES` in
`train_trust/data/bigvul.py` — typically a one-line change.

### Wiring up Joern

```bash
# 1. Install Joern (https://docs.joern.io/installation)
# 2. Verify
joern-parse --help

# 3. Enable in this project (cache becomes invalidated since signals change!)
export USE_JOERN=1
rm -rf data/bigvul/cache  # force re-extraction with Joern
python -m train_trust.scripts.preprocess --config config.yaml
python -m train_trust.train --config config.yaml
```

---

## Reproducing the deck's experiments

| Slide | RQ | What to run |
|---|---|---|
| 13 | RQ1 effectiveness | `python -m train_trust.train` on intra-project BigVul; report F1, AUC, avg IoU, T-score |
| 14 | RQ2 cross-project | Same model, evaluate on MegaVul / SARD / PrimeVul (point `eval` at each) |
| 15 | RQ3 ablation | Set `alpha_attn_iou=0` (or `_syn`, `_pdg`) in config.yaml and re-run |
| 16 | RQ4 sensitivity | Sweep `lambda_trust ∈ {0, 0.1, 0.25, 0.5, 1, 2, 4}` |
| 17 | RQ5 baselines | Compare against `lambda_trust=0` (vanilla), CausalVul, UntrustVul retraining |

A bash sweep helper goes in `train_trust/scripts/sweep.sh` (left as an
exercise — easy `for λ in ...; do python -m train_trust.train --override
trust_loss.lambda_trust=$λ ...`).

---

## What's deliberately NOT included

- **No BigVul download script.** Licensing and storage; build your own loader.
- **No Joern installer.** Java environments vary too much.
- **No GPU-specific optimisations.** Add `accelerate` or DeepSpeed once you
  hit OOM.
- **No real syntax-benign classifiers.** Train them per UntrustVul's
  protocol on historical L+/L− data, then plug into `EnsembleClassifier`.
- **No experiment-tracking.** Pipe the train.py log into wandb/MLflow as
  you prefer.

---

## Where to read first

If you're trying to understand the code before extending it:

1. `train_trust/trust/loss.py` — the novel contribution. Start here.
2. `tests/test_loss.py` — the loss's expected behaviour, made executable.
3. `train_trust/train.py` — how everything is wired together.
4. `train_trust/models/detector.py` — line-level attention extraction
   (the only non-obvious model bit).

Everything else is plumbing.

# Session Recap — Regional Magnitude Pruning (RMP)

**Date:** 2026-04-16
**Project:** `/home/cmoryah/proj/brain-damage/`
**Goal:** Accelerate LLM inference by pruning contiguous weight tile regions and skipping their compute at runtime.

---

## What We Built

### Files
- `config.py` — all experiment parameters (model, tile size, pruning targets, calibration settings)
- `eval_mmlu.py` — MMLU subset evaluation via log-prob scoring (10 subjects, 2028 questions, deterministic)
- `run_baseline.py` — standalone baseline eval script
- `visualize_importance.ipynb` — main experiment notebook (importance scoring, heatmaps, pruning eval)
- `PLAN.md` — full project plan with phases 1-6
- `FINDINGS.md` — chronological decision log with all results
- `requirements.txt` — project dependencies
- `results/` — JSON result files from each experiment run

### Current Config State
```python
MODEL_NAME = "google/gemma-3-1b-it"  # instruction-tuned
TILE_SIZES = [(128, 128)]
PRUNING_TARGETS = [0.10, 0.20, 0.30, 0.50]
PRUNE_TARGETS_PATTERNS = ["gate_proj", "up_proj", "down_proj"]  # MLP only
MAX_PRUNE_PER_MATRIX = 0.50
PRUNE_SKIP_LAYERS = []  # removed — actw naturally protects critical layers
CALIBRATION_SAMPLES = 1024  # doubled from 512
CALIBRATION_SEQ_LEN = 512
```

---

## Key Results

### Baseline
- `google/gemma-3-1b-it`: **30.8%** overall on MMLU subset (well above 25% random)
- `google/gemma-3-1b-pt`: 23.8% (random chance — unusable for measuring degradation)

### Best Pruning Result
- **Activation-weighted at 30% target: 27.4%** (only -3.4pp from baseline)
- Frobenius at 30%: 24.5% (-6.3pp) — much worse

### Full Comparison (50% per-matrix cap, no layer skip)

| Pruning | Frobenius | ActWeight | Winner |
|---|---|---|---|
| 10% | 22.7% | **24.2%** | ActW (+1.5pp) |
| 20% | **23.2%** | 22.9% | Frob (-0.3pp) |
| 30% | 24.5% | **27.4%** | ActW (+2.9pp) |
| 50% | **26.3%** | 22.9% | Frob (-3.4pp) |

---

## Key Decisions & Learnings (chronological)

1. **Base model useless for eval** — pretrained model at random chance, switched to instruct
2. **Raw Frobenius norm fails** — different component types have different scales, `down_proj` gets 100% pruned. Fixed with per-component-type normalization.
3. **Per-matrix z-score too uniform** — erases cross-layer signal. Per-component-type normalization is the right balance.
4. **Per-matrix cap needed** — even with good normalization, outlier matrices can hit 100%. Cap at 50%.
5. **Activation-weighted scoring better than Frobenius** — uses calibration pass (C4 data, forward hooks on MLP layers) to measure actual input activation norms per tile. Score = `||W_tile||_F * mean(||x_inputs||)`.
6. **ActW naturally protects critical layers** — layers 0-4 score highest by both metrics. Explicit layer skip list is redundant for actw (identical results with/without).
7. **Frobenius + layer skip is counterproductive** — forces pruning budget to concentrate in remaining layers.
8. **1B model is very tightly packed** — even 8% effective pruning destroys most signal. Larger models likely have more genuine redundancy.

---

## Known Issues to Fix

1. **Per-layer pruning density chart (section 7)** shows UNCAPPED pruning ratios. It computes `norm_tiles < threshold` without applying `get_capped_mask_viz`. Layers show 90%+ pruning in the chart but actual eval applies the 50% cap. Need to fix the chart to use capped masks.

2. **Histogram (section 8)** still shows Frobenius distribution, should show activation-weighted.

---

## What's Running / Pending

- Just ran notebook with 1024 calibration samples (doubled from 512). Results pending — check if more calibration data improves actw accuracy.
- Layer skip removed (no difference for actw).

---

## Next Steps (from PLAN.md)

1. **Check 1024 calibration results** — does doubling samples help?
2. **Fix visualization bugs** (capped pruning density chart, histogram)
3. **Phase 3: Custom Triton kernel** for block-sparse matmul (actual speedup, not just zeroing)
4. **Phase 5: Layer duplication** — use importance map to identify high-value layers and duplicate them (RYS-style). The actw per-layer importance chart already shows duplication candidates.
5. **Try larger model** — 1B may be too small for meaningful pruning. If hardware allows, test on 4B+ model.
6. **Combine prune + duplicate** — prune dead tiles, duplicate critical layers, keep net params similar.

---

## Hardware
- RTX 4070 Laptop, 8 GB VRAM
- 30 GB RAM, 108 GB disk free
- Python 3.10, PyTorch 2.x, Transformers 4.50.3, Triton 3.2.0

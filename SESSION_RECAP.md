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
- `FINDINGS.md` — detailed chronological decision log with all results and rationale
- `requirements.txt` — project dependencies
- `results/` — JSON result files from each experiment run

### Current Config State
```python
MODEL_NAME = "google/gemma-3-1b-it"  # instruction-tuned
TILE_SIZES = [(128, 128)]            # validated as best granularity
PRUNING_TARGETS = [0.10, 0.20, 0.30, 0.50]
PRUNE_TARGETS_PATTERNS = ["gate_proj", "up_proj", "down_proj"]  # MLP only
MAX_PRUNE_PER_MATRIX = 0.50
PRUNE_SKIP_LAYERS = []               # actw/gradient naturally protect critical layers
CALIBRATION_SAMPLES = 1024
CALIBRATION_SEQ_LEN = 512
```

---

## What We Tried (chronological)

### Models
- **`google/gemma-3-1b-pt` (pretrained):** Scored 23.8% on MMLU — random chance. Unusable for measuring pruning degradation. Pruned models appeared to *improve* over baseline, which was just noise around 25%.
- **`google/gemma-3-1b-it` (instruct):** Scored **30.8%** — clear signal above random. Pruning degradation now measurable.

### Importance Scoring Methods (3 implemented)
1. **Frobenius norm** — `||W_tile||_F`. Cheapest (instant). Worst at identifying expendable tiles. Counterintuitive pattern: accuracy goes UP with more pruning because the per-matrix cap overrides its bad decisions at high sparsity, creating more uniform (less damaging) pruning.
2. **Activation-weighted (ActW)** — `||W_tile||_F * mean(||x_inputs||)`. ~53s calibration (forward only). Better than Frobenius. Naturally protects early layers without needing explicit skip lists. Best at 30% pruning.
3. **Gradient/Taylor** — `Σ|w * ∂L/∂w|`. ~10.5 min calibration (forward + backward). Best at low sparsity (10-20%). Uses ~6.8 GB VRAM during calibration (weights + gradients + saved activations).

### Tile Sizes
- **128x128:** Best. Captures meaningful functional units.
- **64x64:** Worse. 4x more tiles but noisier importance scores. The actw 30% sweet spot collapsed from 28.1% to 22.8%. Frobenius was marginally better at 64x64 (purely local metric benefits from finer resolution) but actw and gradient need the coarser granularity.

### Calibration Data (for ActW and Gradient)
- **512 samples:** ActW 30% = 27.4%
- **1024 samples:** ActW 30% = 28.1% (+0.7pp improvement at the sweet spot)

### Normalization Strategies
- **Raw Frobenius:** Failed — `down_proj` (2.5x smaller weights) got 100% pruned.
- **Per-matrix z-score:** Too uniform — erased cross-layer signal (28-32% per layer).
- **Per-component-type z-score (current):** Right balance — fixes cross-type scale gap while preserving cross-layer variation.

### Layer Protection
- **Skip layers 0-4:** No effect for actw/gradient (they already score early layers highest). Counterproductive for Frobenius (concentrates pruning in remaining layers).

### Per-Matrix Cap
- **70%:** Not protective enough — layers still hit 90%+ when all 3 MLP matrices individually hit cap.
- **50% (current):** Better. At 50% target pruning, 42/78 matrices hit the cap.

---

## Final Results — Three-Way Comparison

Baseline: **30.8%** | Random chance: 25.0% | 128x128 tiles, 1024 cal, 50% cap

| Pruning | Frobenius | ActWeight | Gradient | Best |
|---|---|---|---|---|
| 10% | 22.7% | 23.6% | **27.3%** | Grad |
| 20% | 23.2% | 22.9% | **28.3%** | Grad |
| 30% | 24.5% | **28.1%** | 26.4% | ActW |
| 50% | **26.3%** | 22.4% | 22.5% | Frob |

**Best result: Gradient at 20% → 28.3%** (only -2.5pp from baseline, 11.4% effective sparsity)

Each method has a different sweet spot:
- **Gradient:** 10-20% (loss-aware, finds the safest tiles to remove)
- **ActW:** 30% (activation patterns capture functional structure at moderate sparsity)
- **Frobenius:** 50% (but only because the cap overrides its bad decisions)

---

## Known Issues to Fix

1. **Per-layer pruning density chart (section 7)** shows uncapped pruning ratios — misleading. Needs `get_capped_mask_viz`.
2. **Histogram (section 8)** still shows Frobenius distribution, should show activation-weighted.

---

## Next Steps

1. **Fix visualization bugs** (capped density chart, histogram)
2. **Phase 3: Custom Triton kernel** for block-sparse matmul (actual inference speedup)
3. **Phase 5: Layer duplication** — duplicate high-importance layers (RYS-style)
4. **Combine prune + duplicate** — prune dead tiles, duplicate critical layers, keep net params similar
5. **Try larger model** — 1B may be too tightly packed for meaningful pruning. 4B+ likely has more genuine redundancy.
6. **Hybrid scoring** — could combining gradient (for low sparsity) with actw (for moderate sparsity) give a better single method?

---

## Hardware
- RTX 4070 Laptop, 8 GB VRAM
- 30 GB RAM, ~108 GB disk free
- Python 3.10, PyTorch 2.x, Transformers 4.50.3, Triton 3.2.0

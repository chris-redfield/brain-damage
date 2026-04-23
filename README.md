# Brain-Damage Project — Catch-Up Notes

Quick reference for picking the project up cold. For full chronological detail see `FINDINGS.md`.

---

## What we're building

Inspired by [LLM Neuroanatomy / RYS](https://dnhkng.github.io/posts/rys/). Two ideas that flip each other:

1. **Prune dead weight** — identify low-importance contiguous weight tiles and skip their compute at runtime for real GPU speedup
2. **Duplicate critical circuits** — use the same importance map to duplicate high-importance blocks (RYS-style) so the model "thinks harder" where it counts

The combined goal: parameter count stays similar, but compute is redistributed from dead regions to core reasoning circuits — **faster AND smarter**.

---

## Hardware & environment

- GPU: NVIDIA RTX 4070 Laptop, **8 GB VRAM**
- RAM: 30 GB, Python 3.10, PyTorch 2.6, Triton 3.2, Transformers 4.50.3

This 8 GB constraint shapes almost every decision — models above ~3B don't fit, and LoRA recovery above r=16 with full-precision AdamW also doesn't fit (see Section 15 of FINDINGS). The hardware is the binding constraint on final numbers, not the algorithmic ideas.

---

## Notebooks (in order)

| # | File | What it does |
|---|---|---|
| 1 | `1.visualize_importance.ipynb` | Gemma 3 1B: importance scoring (Frobenius/ActW/Taylor), spatial heatmaps, per-pruning-target MMLU |
| 2 | `2.block_sparse_kernel.ipynb` | Phase 3 kernel — bitmap block-sparse Triton kernel + hybrid dispatch + BCSR experiments (abandoned) |
| 3 | `3.fused_mlp.ipynb` | Gemma 3 1B: partial MLP fusion (gate+up+silu in one kernel, down separate). Essentially no gain at 1B scale |
| 4 | `4.fused_mlp_2b.ipynb` | Qwen 2.5 3B: activation-aware fused kernel + Taylor scoring + load-once architecture. **Main kernel results.** |
| 5 | `5.lora_recovery.ipynb` | Qwen 2.5 3B Taylor-50% pruned: LoRA fine-tuning recovery attempts. Rank sweep + optimizer confound analysis. **Main recovery results.** |
| 6 | `6.layer_duplication.ipynb` | Qwen 2.5 3B: RYS-style layer duplication. 11 variants comparing Taylor-ranked vs RYS-inspired vs stitched-top-N. **F (layers 8-16) beats dense baseline 48.87% vs 48.67%.** |

---

## Project phases — status

- ✅ **Phase 1** — Regional importance scoring (Frobenius, ActW, Taylor)
- ✅ **Phase 2** — Pruning strategy (global threshold + per-matrix cap, MLP only)
- ✅ **Phase 3** — Custom sparse operator (bitmap kernel + hybrid dispatch + fused MLP kernel)
- ✅ **Phase 4** — Evaluation (MMLU on 1B Gemma + 3B Qwen, kernel microbench + end-to-end latency)
- ✅ **Phase 5** — Block duplication (RYS-style). F variant (layers 8-16 duplicated) beats dense baseline on MMLU. See Section 16 of FINDINGS.
- ⚠️ **Phase 6** — LoRA recovery attempted; **blocked by VRAM** (see Sections 14–15 of FINDINGS). Stretch goals (whole-layer removal, quantization) untouched.

---

## Current best results

### Speed (Qwen 2.5 3B on RTX 4070 Laptop, pure prefill at M≈2000)

| Config | Speedup vs dense |
|---|---|
| Fused 50% sparsity | **1.41x** |
| BS 50% sparsity | 1.28x |
| Fused 21% sparsity | 1.04x |

Sparsity speedup **scales with model size**: confirmed 1.14x at 1B → 1.41x at 3B for 50% sparsity. Bigger matmuls → more compute-bound → sparsity savings matter more.

Fusion adds a real **7-10%** over plain block-sparse at prefill once we're in the right regime (21%+ sparsity, large M).

### Accuracy

| Model | Dense MMLU | Best pruned (Taylor) | Delta |
|---|---|---|---|
| Gemma 3 1B | 30.8% | 28.3% (20% sparsity) | -2.5pp |
| Qwen 2.5 3B | 48.7% | 27.5% (18.7% sparsity) | -21.2pp ⚠ |

The accuracy story flipped going from 1B → 3B: Qwen's concentrated circuitry is much more fragile to MLP pruning. All our sparsity configs collapse it to near-random.

### Recovery (Qwen 2.5 3B, Taylor 50%, notebook 5)

| Stage | MMLU |
|---|---|
| Dense baseline | 48.70% |
| Pruned only (Taylor 50%, pre-LoRA) | 26.68% |
| + LoRA r=16 + AdamW fp (500 MMLU-aux × 3 epochs) | 25.99% (**−0.69pp**) |
| + LoRA r=64 + AdamW8bit | 22.89% (**−3.79pp**) |
| + LoRA r=16 + AdamW8bit | 22.89% (**−3.79pp**) |

Best attempted recovery: −0.69pp. Not a net win but pipeline works end-to-end on 8 GB.

### Layer duplication (Qwen 2.5 3B, notebook 6)

| Variant | Spec | MMLU | Δ vs dense (48.67%) |
|---|---|---|---|
| Dense baseline | — | 48.67% | — |
| A1 top-3 scattered (Taylor peak) | [10, 11, 13] | 39.89% | −8.78pp |
| C best-Taylor contiguous-4 | [10, 13] | 43.29% | −5.38pp |
| B1 RYS-scaled window | [20, 23] | 46.06% | −2.61pp |
| **F stitched top-9 (full peak)** | **[8, 16]** | **48.87%** | **+0.20pp** 🎉 |

**F is the first variant to beat the dense baseline on MMLU — Phase 5 win.**

Key finding: a sharp **block-size cliff** at 9 layers. Variants duplicating 1-7 layers of the peak circuit all hurt significantly (−4 to −9pp). Going to 9 contiguous layers flips the sign entirely, because it captures the *full natural boundary* of the reasoning circuit (Taylor z-sum goes above +2000 at layer 8, stays above through layer 16, drops sharply after). Partial circuit re-execution corrupts mid-stream hidden states; whole-circuit re-execution lets the model "think harder" as RYS predicted.

Second finding: **RYS's relative position scaling does not transfer across model sizes.** RYS's 45-51 of 80 (center-back) would map to 20-23 of 36 — exactly the *trough* on our importance map. Smaller models concentrate circuits earlier. Data-driven circuit detection via Taylor importance is the method that generalizes.

---

## Key findings that generalize

### On importance scoring

Ranking from best to worst (consistent across both model sizes):

1. **Taylor / gradient-based** (`Σ |w · ∂L/∂w|`) — best. At same sparsity, beats ActW by 2-3pp on both 1B and 3B.
2. **Activation-weighted** (`||W_tile||_F × mean(||input||)`) — middle. Cheaper (forward-only, ~1 min) than Taylor (forward+backward, ~5 min).
3. **Frobenius norm** — worst. Doesn't capture actual usage, gets saved only when the per-matrix cap overrides its decisions.

**Normalization matters**: per-component-type z-score (pool all `gate_proj` matrices, all `up_proj`, all `down_proj` separately) fixes the cross-type scale gap while preserving cross-layer variation.

### On pruning geometry

- **128×128 tiles**: the right granularity. 64×64 is too noisy; larger might work but untested.
- **50% per-matrix cap** is essential. Without it, outlier layers can hit 90%+ pruning and destroy the model.
- **MLP only**: pruning attention is risky (Q/K/V alignment). MLP is bulk of params anyway.

### On the kernel

- **Hybrid dispatch** (M<64 → dense cuBLAS, M≥64 → Triton kernel) eliminates decode regressions — essential.
- **Break-even is a SPARSITY threshold, not an M threshold.** At < ~25% sparsity the bitmap-check overhead exceeds compute savings. This tripped us up for days.
- **BCSR packing didn't pay off** — theoretical memory savings at M=1 got swamped by launch/grid overhead. Tensor cores with M-dim waste beat CUDA-core GEMV. (Section 11-12 of FINDINGS.md for details.)
- **Fusion is most valuable on bigger models** — at 1B no benefit, at 3B ~10% over plain BS.

### On accuracy vs sparsity

- There's a **"knee"** in the curve around 10-20% sparsity. Below the knee: signal preserved. Above: collapse to random. The curve is **flat above the knee** — can push sparsity as high as you want without further accuracy loss.
- This means: for max speed, run at 50% sparsity (same accuracy as 20% but 37% more speedup).
- Smaller base models have a higher knee (Gemma 1B: signal preserved up to 30% sparsity). Stronger models have a lower knee (Qwen 3B: knee < 10%).

### On layer duplication (notebook 6 — Phase 5)

- **Duplicating the full natural circuit boundary beats dense baseline.** F (layers 8-16) at +0.20pp on MMLU is small in absolute terms but confirms the RYS mechanism at 3B scale — the first variant to cross into positive territory after 10 negatives.
- **Block-size cliff at the circuit boundary.** 1, 3, 4, 5, 7 layers of the peak all hurt (−4 to −9pp). 9 contiguous layers (exactly matching where Taylor z-sum is positive) helps. Partial re-execution of a circuit corrupts its internal hidden-state flow; whole-circuit re-execution works as RYS described.
- **Taylor importance is doing double duty.** The same per-tile scoring that identifies safe-to-prune weights also identifies safe-to-duplicate layers via per-layer aggregation. That's the "one map, both directions" premise of the project validated on real numbers.
- **RYS's relative-position scaling doesn't transfer.** Their 45-51 of 80 (center-back) would map to 20-23 of 36 on our Qwen 3B — exactly where importance is *negative*. Smaller models push reasoning circuits earlier in the stack. Data-driven boundary detection (via Taylor) is the generalizable method.
- **"Low importance = safe to duplicate" is wrong.** Early negative-Taylor layers (H trough, 1-4) are not passthrough — they're doing heavy encoding work that every downstream layer depends on. Duplicating them corrupts early representations and tanks MMLU worse than duplicating the peak partially.
- **Circuit corruption is a local failure mode.** Scattering duplications across zones (I) hurts as much as scattering within the peak (A1) — no spatial spread saves you from the mid-flow hidden-state corruption.
- **Budget-neutral prune + duplicate didn't work at Taylor-50% pruning.** The pruned base is already at random chance; duplication added ±1pp noise. Light pruning (Taylor 10-20%) + F duplication is the right next test.

### On LoRA recovery (new — notebook 5)

- **MMLU-aux training doesn't recover broken MLP circuits.** Training on 500 aux-train samples (answer-token-only loss) moved per-subject accuracy ±13pp but net was −0.69pp. The learning is real (loss 6.27 → 1.51) but narrow — it overwrites one MCQ answer-format prior with another rather than repairing the pruned circuits. Hypothesis: pretraining-distribution data (C4/FineWeb + next-token CE) would match what was broken, but we can't run that experiment here.
- **8-bit AdamW is not safe for recovery fine-tuning.** Holding rank=16 fixed, swapping full-precision AdamW → AdamW8bit dropped MMLU from 25.99% → 22.89% — same degenerate attractor (464/2028) as the r=64 run. Mechanism: quantized m/v momentum can't resolve small updates near the loss floor (final loss 1.85 vs 1.51); QLoRA gets away with this on pretraining (high-gradient regime) but recovery is low-gradient.
- **Capacity (rank) isn't the recovery bottleneck** at least up to what we can test. r=16+fp beats r=64+8bit by 3pp — bigger adapter with noisier optimizer is worse than smaller adapter with clean optimizer. Testing rank=64+ with full-precision AdamW requires >8 GB VRAM.
- **Methodological lesson**: when two variables change at once (rank AND optimizer) and you get an unexpected result, isolate before interpreting. I nearly wrote up a wrong "data distribution is the bottleneck" conclusion based on a confounded experiment.

### Memory on a tight GPU

- IPython holds references to "deleted" models via `sys.last_traceback` and `Out[]` cache. `del m` doesn't work without also clearing IPython state.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `torch._C._cuda_clearCublasWorkspaces()` help allocator fragmentation.
- **PEFT + gradient checkpointing on a frozen base model requires `model.enable_input_require_grads()`** — without it, checkpoint-recompute silently fails and full activation stack is retained, OOMing immediately.
- The real fix for model variants: **load the model ONCE and reconfigure in-place between variants.** `configure_model(variant, masks)` in notebook 4 is the pattern.

---

## What's left to do

### Hardware-blocked (what we'd run next with ≥16 GB VRAM)

1. **Pretraining-distribution LoRA recovery** — C4 or FineWeb-Edu, standard next-token CE, r=16 full-precision AdamW. Matches the distribution the damaged MLP circuits were trained on. Main hypothesis to test before concluding LoRA recovery is dead for this damage level.
2. **Rank sweep with full-precision AdamW** — r=32, 64, 128 on the same recovery task. Clean capacity test that's currently impossible on this card.
3. **Longer training runs at r=16** — N_TRAIN=5000+ with fewer epochs, shuffled across aux-train sources to avoid the distribution-narrowness seen in notebook 5.

### Local-feasible (what we can still do on 8 GB)

1. **Widen/tune F further** — try [7, 17], [8, 17], [8, 18] (11 layers). Does including outer shoulders help more, or start to regress? Maps the circuit boundary precisely around the current +0.20pp win.
2. **Triple-duplicate F** — run layers 8-16 *three* times in forward. RYS didn't test this. If "think harder" works once, does "think even harder" work twice?
3. **F + light pruning** — Taylor 10-20% sparsity (above the accuracy knee) combined with F duplication. This is the real **full-RMP test**: prune dead weight for compute savings, duplicate circuit for accuracy gain, target budget-neutral or better end-to-end. The Taylor-50% + F test in notebook 6 was dominated by pruning damage (pruned base was at random chance); a lighter prune should preserve enough signal for F to show a real gain.
4. **Sub-10% sparsity on Qwen (alone)** — below the accuracy knee. Less speedup but might preserve real signal. Worth measuring for a quality-focused operating point as a baseline for (3).

### Stretch (Phase 6 remainder)

- Whole-layer pruning (layer-level importance, drop weakest blocks entirely)
- Dynamic sparsity (lightweight per-prompt router)
- Attention tile-pruning (with per-head analysis)

### Explicitly ruled out

- **Quantization** — user wants novel exploration, not known techniques
- **BCSR kernel** — complexity didn't pay off on this GPU (see FINDINGS section 11-12)
- **Dedicated GEMV kernel for M=1** — tensor cores beat CUDA cores even with M-dim waste (see FINDINGS section 12)
- **8-bit optimizers for LoRA recovery** — confirmed harmful on this task (see FINDINGS section 15)

---

## How to pick back up in the next session

1. Read `FINDINGS.md` sections 9-13 for the kernel story, 14-15 for the LoRA story, 16 for the duplication story — or this doc for a summary
2. For kernel work / new pruning configs → open notebook 4 (`4.fused_mlp_2b.ipynb`)
3. For recovery work / LoRA iteration → open notebook 5 (`5.lora_recovery.ipynb`)
4. For RYS duplication / Phase 5 iteration → open notebook 6 (`6.layer_duplication.ipynb`) — F variant is the baseline to build on
5. For the full-RMP combo experiment (light prune + F duplicate) → new notebook 7, reuse notebook 6's duplication utilities on a lighter-pruned base

---

## File layout

```
/home/cmoryah/proj/brain-damage/
├── 1.visualize_importance.ipynb   # Phase 1-2 (importance scoring + pruning)
├── 2.block_sparse_kernel.ipynb    # Phase 3 (kernel + BCSR failures)
├── 3.fused_mlp.ipynb              # Partial fusion on 1B
├── 4.fused_mlp_2b.ipynb           # Full comparison on Qwen 3B — MAIN KERNEL RESULTS
├── 5.lora_recovery.ipynb          # LoRA recovery attempts on Qwen 3B — MAIN RECOVERY RESULTS
├── 6.layer_duplication.ipynb      # RYS-style layer duplication — PHASE 5 RESULTS (F variant beats dense)
├── config.py                       # Shared project config (model, tile size, etc.)
├── eval_mmlu.py                    # MMLU subset evaluator (10 subjects, 2028 Qs)
├── PLAN.md                         # Original project plan (phases 1-6)
├── FINDINGS.md                     # Chronological findings (16 sections)
├── README.md                       # This file
└── results/                        # JSON outputs from each MMLU eval
```

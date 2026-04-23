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
| 6 | `6.layer_duplication.ipynb` | Qwen 2.5 3B: RYS-style layer duplication. 14 variants. **No narrow duplication beats dense.** Wide duplication (layers 9-34, 72% of stack) lands at 48.87% vs dense 48.67% — within noise (Phase 5 negative result). See FINDINGS §17. |

---

## Project phases — status

- ✅ **Phase 1** — Regional importance scoring (Frobenius, ActW, Taylor)
- ✅ **Phase 2** — Pruning strategy (global threshold + per-matrix cap, MLP only)
- ✅ **Phase 3** — Custom sparse operator (bitmap kernel + hybrid dispatch + fused MLP kernel)
- ✅ **Phase 4** — Evaluation (MMLU on 1B Gemma + 3B Qwen, kernel microbench + end-to-end latency)
- ⚠️ **Phase 5** — Block duplication (RYS-style) attempted across 14 variants. **Negative result**: narrow duplications (1-11 layers) hurt MMLU by 3-14pp; wide duplication (layers 9-34, 72% of stack) is within noise of dense. Earlier "F=[8,16] beats dense by +0.20pp" claim was a misread — real F is [9,34]. See FINDINGS §17 for the correction.
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

### Layer duplication (Qwen 2.5 3B, notebook 6) — Phase 5 negative result

| Variant | Spec | Dup size | MMLU | Δ vs dense (48.67%) |
|---|---|---|---|---|
| Dense baseline | — | — | 48.67% | — |
| A1 top-3 scattered (Taylor peak) | [10, 11, 13] | 3 | 39.89% | −8.78pp |
| C best-Taylor contiguous-4 | [10, 13] | 4 | 43.29% | −5.38pp |
| B1 RYS-scaled window | [20, 23] | 4 | 46.06% | −2.61pp |
| [8, 16] (the one I mistakenly called "F") | — | 9 | 38.21% | −10.46pp |
| F_L / F_R / F_LR widenings around [8, 16] | — | 10-11 | 34-40% | −9 to −14pp |
| **F stitched top-9 (actual)** | **[9, 34]** | **26** | **48.87%** | **+0.20pp (within noise)** |

**No duplication variant beat dense with statistical significance.** +0.20pp over 2028 MMLU questions is ±4 questions — within the ~±1.1pp standard-error noise floor.

Key observed pattern:
- **Narrow duplications (1-11 layers) consistently hurt** by 3-14pp.
- **Wide duplication (26 layers, 72% of the model) is approximately neutral** — which makes sense physically: duplicating most of the model approaches "run the model twice", preserving computation consistency by symmetry.
- **No tight "circuit window" was found that beats dense.** The earlier claim of a "razor-sharp boundary at [8, 16] with +0.20pp" was a **misread** of the variants output (see FINDINGS §17 for the correction). The actual winning F was a 26-layer, 72%-of-stack duplication.

Still valid observation: **RYS's position scaling from 72B does not transfer to 3B straightforwardly.** Relative scaling of their 45-51 of 80 would put the window at 20-23 of 36 — in the trough of our Taylor importance map.

What this means for the project thesis: the "duplicate critical circuits" half of the Taylor map's double-duty hasn't been demonstrated at this model size. Pruning half (prune dead weight) is well-validated; duplication half is an open negative result.

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

### On layer duplication (notebook 6 — Phase 5 negative result)

- **No duplication variant we tested beats dense with statistical significance.** +0.20pp on F=[9, 34] is within the MMLU ±1.1pp noise floor. Phase 5 should be marked as an attempted experiment with a negative result, not a win.
- **Narrow local duplications universally hurt** (3-14pp regressions across 13 variants with dup sizes from 1 to 11 layers). The mechanism is plausible: duplicating a narrow mid-stream block feeds out-of-distribution hidden states to downstream layers, corrupting computation. But this isn't validated as a *useful* result — it's just "narrow local duplication is bad."
- **Wide duplication (26 layers, 72% of the stack) is approximately neutral.** This makes sense by symmetry — duplicating most of the model approaches "run the model twice" with consistent computation.
- **RYS's relative-position scaling doesn't transfer.** Their 45-51 of 80 (center-back) would map to 20-23 of 36 on our Qwen 3B — exactly where importance is *negative*. Smaller models push reasoning circuits earlier in the stack. Still a valid observation.
- **"Low importance = safe to duplicate" is wrong** (H trough [1, 4] at −8.53pp). Early negative-Taylor layers are doing heavy encoding work that downstream layers depend on, not being passthrough.
- **Scattered duplications across zones hurt as much as scattered within the peak** (I gap vs A1) — scattering itself is the destructive pattern.
- **Taylor "double duty" is NOT validated at the duplication end.** The importance map tells us what's safe to *prune* (well-validated). It does not, on this evidence, reliably tell us what's beneficial to *duplicate* at this model size.
- **Methodological lesson.** An earlier version of FINDINGS Section 16 built a strong "razor-sharp boundary at [8, 16]" narrative and a "block-size cliff" story based on a misread of the variants-output cell. F was always range [9, 34], not [8, 16]. The adjacent "widening" variants (F_L, F_R, F_LR) weren't widening the winner — they were nearby narrow windows, and their collapse was the same pattern as every other narrow window in the sweep. Always read the variant spec from code/logs, not from chart interpretation.

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

1. **Sweep narrow duplication windows systematically** — all [start, end] pairs with end − start ∈ {3, 5, 7, 9} across the stack. The current evidence is that narrow duplications hurt everywhere we tested; but we only sampled ~10 configs. A more complete sweep would verify there's no hidden sweet spot RYS-style. Cheap: 3240 configs like RYS's sweep = too many; a 36×36/2 structured sweep = 648 configs × 90s = too long; but e.g. fixed-size-7 window across 30 positions = 30 evals × 90s = 45 min.
2. **Sub-10% sparsity on Qwen** — below the accuracy knee. Less speedup but might preserve real signal. Worth measuring for a quality-focused operating point.
3. **Pruning + LoRA recovery with pretraining distribution OR more data** — notebook 5's MMLU-aux approach didn't work. Try C4 streaming + next-token CE at r=16 with full-precision AdamW. (VRAM-constrained; maybe with MAX_LEN 96 it fits.)
4. **Attention tile-pruning** (Phase 6 stretch) — currently excluded. Per-head analysis + MLP-style tile pruning within Q/K/V projections. Risky but unexplored.

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

1. Read `FINDINGS.md` sections 9-13 for the kernel story, 14-15 for the LoRA story, 16 for the initial (partially retracted) duplication story, **17 for the correction and Phase 5 negative result**. Or this doc for a summary.
2. For kernel work / new pruning configs → open notebook 4 (`4.fused_mlp_2b.ipynb`)
3. For recovery work / LoRA iteration → open notebook 5 (`5.lora_recovery.ipynb`)
4. For RYS duplication / Phase 5 iteration → open notebook 6 (`6.layer_duplication.ipynb`). **Caveat**: the winning F variant is range [9, 34] (26-layer dup) and its +0.20pp is within noise — there is no validated narrow "circuit window" that beats dense.
5. Notebook 7 (`7.rmp_full.ipynb`) exists but its original F=[8, 16] assumption was wrong. The `F_RANGE` constant has been corrected to (9, 34), matching notebook 6's real winning variant. Re-running it would sweep Taylor pruning levels with the 26-layer duplication applied — but given F is within noise, the test is effectively "pruning only" with extra compute overhead. Useful if you want the pruning-sweep data on Qwen 3B, not if you expect a duplication-driven gain.

---

## File layout

```
/home/cmoryah/proj/brain-damage/
├── 1.visualize_importance.ipynb   # Phase 1-2 (importance scoring + pruning)
├── 2.block_sparse_kernel.ipynb    # Phase 3 (kernel + BCSR failures)
├── 3.fused_mlp.ipynb              # Partial fusion on 1B
├── 4.fused_mlp_2b.ipynb           # Full comparison on Qwen 3B — MAIN KERNEL RESULTS
├── 5.lora_recovery.ipynb          # LoRA recovery attempts on Qwen 3B — MAIN RECOVERY RESULTS
├── 6.layer_duplication.ipynb      # RYS-style layer duplication — PHASE 5 (negative result, see FINDINGS §17)
├── 7.rmp_full.ipynb               # Full RMP combo (prune + duplicate + LoRA); F_RANGE corrected to (9, 34)
├── config.py                       # Shared project config (model, tile size, etc.)
├── eval_mmlu.py                    # MMLU subset evaluator (10 subjects, 2028 Qs)
├── PLAN.md                         # Original project plan (phases 1-6)
├── FINDINGS.md                     # Chronological findings (17 sections; §17 corrects §16)
├── README.md                       # This file
└── results/                        # JSON outputs from each MMLU eval
```

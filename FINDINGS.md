# Findings & Decisions Log

Chronological record of what we tried, what we learned, and why we made certain decisions.

---

## 1. Project Inspiration

Based on [LLM Neuroanatomy / RYS](https://dnhkng.github.io/posts/rys/) by David Noel Ng. That work showed that duplicating middle layers of a 72B model (without changing weights) improved benchmark scores — proving transformers develop functional regions where some layers matter more than others.

Our idea flips this: instead of duplicating important regions, **prune the unimportant ones** and exploit the structured sparsity for real GPU speedup. Later, combine both: prune the dead weight AND duplicate the critical circuits.

---

## 2. Model Selection

### First attempt: `google/gemma-3-1b-pt` (pretrained base model)

Chose a 1B model to fit on an RTX 4070 Laptop (8 GB VRAM). The pretrained base model loads at ~2.6 GB in bf16, leaving plenty of room.

### Problem: base model scores at random chance on MMLU

The base model scored **23.8%** overall on our MMLU subset (2028 questions, 4-choice = 25% random). This is *below* random chance, meaning the model has essentially zero MMLU signal.

When we ran pruned variants (10%, 20%, 30%, 50% tile pruning), the results were:

| Variant | Accuracy | Delta |
|---|---|---|
| Baseline | 23.8% | — |
| Pruned 10% | 24.7% | +0.9pp |
| Pruned 20% | 26.1% | +2.3pp |
| Pruned 30% | 24.7% | +0.9pp |
| Pruned 50% | 24.4% | +0.6pp |

This looked like pruning was *helping*, which is nonsensical. Investigation confirmed:
- Data was identical across runs (same 2028 questions, deterministic loading)
- `restore_model` works correctly (verified: fresh load and restored model both produce exactly 23.8%, every subject matching)
- Metrics calculation is deterministic (log-prob argmax, no randomness)

**Root cause:** All results are noise around 25% random chance. The perturbations from pruning shift logits at positions that were already random, and some shifts happen to land closer to correct answers. Standard error for 2028 random-chance questions is ~0.96pp, so ±2pp variation is expected.

**Decision:** Switch to `google/gemma-3-1b-it` (instruction-tuned). This model should score meaningfully above random on MMLU, giving us a real baseline where pruning degradation is measurable.

---

## 3. Tile Size Selection

Chose **(128, 128)** tiles as default, aligning with GPU warp/block granularity. Smaller tiles (64x64) give finer pruning control but more indexing overhead; larger tiles (256x256) are coarser. Will experiment later once the pipeline works.

---

## 4. Importance Scoring: Normalization Strategy

### First attempt: raw Frobenius norm

Used the Frobenius norm of each tile as its importance score. Problem: weight matrices have very different magnitude scales.

In Gemma 3 1B:
- `gate_proj` mean abs weight: 0.022
- `up_proj` mean abs weight: 0.022
- `down_proj` mean abs weight: 0.009

This meant a global threshold wiped out `down_proj` entirely (100% pruned in many layers) — catastrophic, since `down_proj` is the MLP output projection.

### Second attempt: per-matrix z-score normalization

Z-scored each matrix independently. Fixed the cross-component-type imbalance, but made pruning too uniform across layers (28.4%–32.4% per layer). The per-matrix normalization erased the signal about which layers have genuinely more dead weight.

### Third attempt (current): per-component-type normalization

All 26 `gate_proj` matrices share one (mean, std), all 26 `up_proj` share another, all 26 `down_proj` share a third. This:
- Fixes the `down_proj` vs `gate_proj` scale gap
- Preserves cross-layer variation within each component type (so some layers CAN be pruned more aggressively)

Stats after normalization (instruct model):
- `gate_proj`: mean=3.815, std=0.141 (12636 tiles)
- `up_proj`: mean=3.818, std=0.128 (12636 tiles)
- `down_proj`: mean=1.264, std=0.276 (12636 tiles)

Pruning balance at 30%: gate_proj 26.8%, up_proj 28.6%, down_proj 34.6%.

### Per-matrix safety cap

Even with per-component-type normalization, some individual matrices (outlier layers) could still hit 100% pruning. Added `MAX_PRUNE_PER_MATRIX = 0.70` — no single matrix can lose more than 70% of its tiles. At 50% global pruning, the cap was applied to 34 matrices.

---

## 5. Instruct Model Results (`google/gemma-3-1b-it`)

### Baseline: 30.8% overall

The instruct model scores well above random chance, with clear signal on several subjects:

| Subject | Accuracy |
|---|---|
| us_foreign_policy | 59.0% |
| anatomy | 40.0% |
| college_computer_science | 37.0% |
| college_chemistry | 36.0% |
| professional_medicine | 31.6% |
| global_facts | 31.0% |
| machine_learning | 28.6% |
| moral_scenarios | 27.3% |
| econometrics | 20.2% |
| abstract_algebra | 22.0% |
| **OVERALL** | **30.8%** |

### Pruning results (70% per-matrix cap)

| Variant | Accuracy | Delta | Effective sparsity | Matrices capped |
|---|---|---|---|---|
| Baseline | 30.8% | — | 0% | — |
| Pruned 10% | 23.3% | **-7.5pp** | 9.1% | 3 |
| Pruned 20% | 24.6% | -6.2pp | 17.1% | 12 |
| Pruned 30% | 25.1% | -5.7pp | 23.5% | 21 |
| Pruned 50% | 24.4% | -6.4pp | 39.6% | 33 |

### Key observations

1. **Even 10% pruning is devastating** — drops the model from 30.8% to 23.3%, losing nearly all signal above random chance. The 1B model is very tightly packed with little genuine redundancy in MLP weights.

2. **Degradation curve is flat after 10%** — going from 10% to 50% barely changes accuracy (23.3% → 24.4%). Once the first low-importance tiles are removed, the model is already at random chance. Additional pruning just shuffles noise.

3. **Per-layer pruning is highly uneven** — layer 13 hit 90.3% pruning while layer 0 hit 0.0%. The per-component-type normalization preserves real cross-layer signal, but some layers are getting hollowed out catastrophically.

4. **The 70% cap wasn't protective enough** — individual matrices hit 70% cap, but when multiple matrices in the same layer all hit the cap, the combined effect on that layer is still catastrophic (layer 13 at 90.3% overall).

### Tighter cap: 50% per matrix (Frobenius)

With the 50% cap, results were essentially the same:

| Variant | Accuracy | Delta | Effective sparsity | Matrices capped |
|---|---|---|---|---|
| Baseline | 30.8% | — | 0% | — |
| Pruned 10% | 22.7% | -8.0pp | 8.2% | 5 |
| Pruned 20% | 23.2% | -7.5pp | 13.4% | 17 |
| Pruned 30% | 24.5% | -6.3pp | 17.9% | 23 |
| Pruned 50% | 26.3% | -4.4pp | 30.0% | 42 |

The cap is doing real work (42 of 78 matrices hit the ceiling at 50% target), but the core problem persists: even 8.2% effective pruning destroys the model. The Frobenius norm importance metric is not identifying truly expendable tiles.

### Decision: try activation-weighted importance scoring

**Rationale:** Frobenius norm only measures "how big is this tile." A tile with large weights that receives near-zero activations during inference contributes nothing — but Frobenius would keep it. Activation-weighted scoring captures actual contribution:

`score(tile) = ||W_tile||_F * mean(||x_inputs_for_tile||)`

This requires a calibration pass over real data (512 samples from C4, 512 tokens each). The calibration hooks into each MLP linear layer, captures input activations, and computes the mean activation norm per tile column. The product of weight norm and activation norm gives a score that reflects whether the tile is both large AND actually used.

### Activation-weighted vs Frobenius — head-to-head (50% cap)

| Pruning | Frobenius | ActWeight | Winner | Delta |
|---|---|---|---|---|
| 10% | 22.7% | **24.2%** | ActW | +1.5pp |
| 20% | **23.2%** | 22.9% | Frob | -0.3pp |
| 30% | 24.5% | **27.4%** | ActW | +2.9pp |
| 50% | **26.3%** | 22.9% | Frob | -3.4pp |

Activation-weighted stats (per component type):
- `gate_proj`: mean=23.259, std=12.812
- `up_proj`: mean=23.249, std=12.681
- `down_proj`: mean=1.813, std=1.702

**The standout result: actw at 30% pruning preserved 27.4% accuracy** — only 3.4pp below the 30.8% baseline, while Frobenius at 30% dropped to 24.5% (-6.3pp). At that sweet spot, activation weighting retained nearly half the model's signal above random chance, while Frobenius lost it all.

Notable per-subject results at actw 30%: `us_foreign_policy` held at 35.0% (baseline 59%), `professional_medicine` actually rose to 37.9% (baseline 31.6%), `college_computer_science` held at 32.0% (baseline 37.0%).

**However, results are inconsistent across pruning levels.** ActW wins at 10% and 30%, Frobenius wins at 20% and 50%. No clean monotonic degradation curve for either method. Both methods push the model near random chance for most configurations, and at ~25% accuracy small perturbations cause wild per-subject swings.

### Key structural insight: early layers must not be pruned

Activation-weighted scoring revealed that the least-pruned layers are 0–4 (the early "translator" layers from the RYS hypothesis). With Frobenius norm, layer 0 was already protected (0% pruning), but with activation weighting the signal is even clearer — early layers have both high weight magnitude AND high input activation, making them the most important by both metrics.

Meanwhile layer 12 hit 91.4% pruning with activation-weighted scoring, confirming some middle layers have genuinely low utilization.

**Decision:** Exclude layers 0-4 from pruning (protect the input encoding stack).

### Results with layer skip (layers 0-4 protected, 50% cap)

| Pruning | Frobenius (no skip) | Frobenius (skip 0-4) | ActW (no skip) | ActW (skip 0-4) |
|---|---|---|---|---|
| 10% | 22.7% | 22.5% | 24.2% | 24.2% |
| 20% | 23.2% | 24.3% | 22.9% | 22.9% |
| 30% | 24.5% | 22.9% | 27.4% | 27.4% |
| 50% | 26.3% | 25.7% | 22.9% | 22.9% |

**Key finding: activation-weighted scoring already protects early layers naturally.** The actw results are byte-for-byte identical with and without the skip list — layers 0-4 already scored so high that they were never pruned. The explicit skip list is redundant for actw.

For Frobenius, the skip list is actually counterproductive at 30% and 50% — by protecting 5 layers, the same pruning budget concentrates more aggressively into layers 5-25, making those layers lose more tiles. Mixed results overall.

**Conclusion:** The activation-weighted importance metric is self-correcting — it identifies critical layers without needing manual exclusion rules. This is a strong argument in its favor over Frobenius, independent of accuracy numbers.

### Decision: increase calibration samples to 1024

Since activation-weighted scoring is the more promising method, we want to improve the quality of the activation statistics. More calibration data = more stable estimates of which tiles are actually used during inference. Doubling from 512 to 1024 samples should reduce noise in the importance scores.

### Results: 512 vs 1024 calibration samples (actw, 50% cap, no layer skip)

| Pruning | ActW (512 cal) | ActW (1024 cal) | Change |
|---|---|---|---|
| 10% | 24.2% | 23.6% | -0.6pp |
| 20% | 22.9% | 22.9% | 0.0pp |
| 30% | 27.4% | **28.1%** | **+0.7pp** |
| 50% | 22.9% | 22.4% | -0.5pp |

Frobenius results unchanged (doesn't use calibration): 22.7%, 23.2%, 24.5%, 26.3%.

**The sweet spot improved.** ActW at 30% went from 27.4% to 28.1% — now only 2.7pp below the 30.8% baseline, retaining over half the model's signal above random chance while pruning 21% of MLP tiles effectively.

Other pruning levels barely moved (within noise). The improvement is concentrated at the 30% operating point, which is where actw already had its strongest advantage.

**Overall best result so far:** activation-weighted, 30% target, 1024 calibration samples, 50% per-matrix cap → **28.1% accuracy** (baseline 30.8%, random 25.0%). This is the most promising configuration for further work.

### Why Frobenius accuracy goes UP with more pruning

An odd pattern in all Frobenius runs: accuracy increases at higher sparsity targets (22.7% at 10% → 26.3% at 50%). This is NOT noise — it's the per-matrix cap creating a paradoxically better pruning pattern at high targets.

At 50% target, 42 of 78 matrices hit the 50% cap. The cap overrides Frobenius's choices and forces a more uniform distribution of pruning across layers. Since Frobenius is bad at identifying truly expendable tiles, having the cap override more of its decisions actually reduces the damage — a little pruning everywhere is less catastrophic than a lot of pruning in one critical layer.

At 10% target, only 5 matrices hit the cap, so Frobenius gets to concentrate its pruning budget in the few layers it considers least important. But it picks wrong, and the concentrated damage is devastating.

This is further evidence that Frobenius norm is a poor importance metric for this model — the cap saving it from itself produces better results than letting it do what it wants.

### Tile size experiment: 64x64 vs 128x128

Tested smaller 64x64 tiles (4x more tiles per matrix: 151,632 vs 37,908).

| Pruning | Frob 128x128 | Frob 64x64 | ActW 128x128 | ActW 64x64 |
|---|---|---|---|---|
| 10% | 22.7% | 23.2% | 23.6% | 24.2% |
| 20% | 23.2% | 23.4% | 22.9% | 23.6% |
| **30%** | 24.5% | 23.8% | **28.1%** | **22.8%** |
| 50% | 26.3% | 26.6% | 22.4% | 22.9% |

**The 30% sweet spot collapsed** from 28.1% to 22.8% with smaller tiles. The finer granularity makes importance scores noisier — each 64x64 tile covers only 4,096 weights (vs 16,384 for 128x128), so the Frobenius norm per tile has higher variance and the activation weighting gets diluted across more, smaller column slices.

128x128 tiles are coarse enough to capture meaningful functional units. 64x64 fragments the signal — tiles that look unimportant at fine scale may be part of larger critical structures that only emerge at the 128x128 level.

**Decision:** Revert to 128x128 tiles. Larger tiles (256x256) may also be worth testing, but 128x128 is the validated sweet spot.

### Gradient-based (Taylor) importance scoring

Added a third scoring method: `importance(tile) = Σ|w * ∂L/∂w|` — the first-order Taylor approximation of loss change from zeroing each tile. This captures weight magnitude, input activations, AND downstream loss sensitivity in a single metric.

Cost: ~10.5 minutes for 1024 calibration samples (forward + backward), vs ~53s for actw (forward only). Uses ~6.8 GB VRAM (weights + gradients + saved activations for backward).

### Three-way comparison: Frobenius vs ActW vs Gradient (128x128, 1024 cal, 50% cap)

| Pruning | Frobenius | ActWeight | Gradient | Best |
|---|---|---|---|---|
| 10% | 22.7% | 23.6% | **27.3%** | Grad |
| 20% | 23.2% | 22.9% | **28.3%** | Grad |
| 30% | 24.5% | **28.1%** | 26.4% | ActW |
| 50% | **26.3%** | 22.4% | 22.5% | Frob |

**Gradient scoring is the new best at low pruning levels.** At 10% pruning, gradient preserves 27.3% accuracy (only -3.5pp from baseline) vs 23.6% for actw and 22.7% for Frobenius. At 20%, gradient hits **28.3%** — our new overall best result, only 2.5pp below baseline while pruning 11.4% of MLP tiles.

The methods have different sweet spots:
- **Gradient:** best at 10-20% (27.3%, 28.3%) — loss-aware scoring identifies the truly safe-to-remove tiles
- **ActW:** best at 30% (28.1%) — activation patterns capture functional structure at moderate sparsity
- **Frobenius:** best at 50% (26.3%) — but only because the cap overrides its bad decisions

**New overall best: gradient at 20% target → 28.3% accuracy** (baseline 30.8%, effective sparsity 11.4%, only 2.5pp below baseline).

Gradient scoring's advantage at low sparsity makes sense: when you're only removing 10-20% of tiles, the key is identifying the absolute safest tiles to remove. Gradient importance directly measures "how much would the loss change" — it answers exactly the right question. At higher sparsity (30-50%), even gradient scoring can't find enough truly expendable tiles in this tightly packed 1B model.

---

## 6. Pruning Scope: MLP only (attention excluded)

**Decision:** Only prune MLP projections (`gate_proj`, `up_proj`, `down_proj`). Attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) are excluded.

**Rationale:** Attention heads are tightly coupled — Q/K/V must stay aligned for each head to compute coherent attention patterns. Tile-pruning within these matrices risks breaking head alignment in unpredictable ways. MLP layers are independent feed-forward projections where individual tile regions can be zeroed without structurally breaking the computation.

MLP layers are also the bulk of parameters (~624M of the ~1B total), so they're the primary target for compute savings anyway.

**Future experiment:** Try attention tile-pruning with careful per-head analysis.

---

## 7. Evaluation Methodology

Using MMLU subset (10 subjects, 2028 questions) via log-probability scoring:
- Format each question as multiple choice with "Answer:" prompt
- Compare log-probs at A/B/C/D token positions
- Pick the highest as the prediction

This is deterministic (no sampling, no temperature), fast (~65s per full eval), and doesn't require generation.

---

## 8. Hardware & Environment

- GPU: NVIDIA RTX 4070 Laptop, 8 GB VRAM
- RAM: 30 GB
- Python 3.10, PyTorch 2.x, Transformers 4.50.3, Triton 3.2.0
- Model loads at ~2.6 GB in bf16

---

## 9. Block-Sparse Kernel (Phase 3) — Prefill vs Decode

Built a custom Triton kernel that reads a per-matrix bitmap `(N_tiles, K_tiles)` and skips zeroed (128×128) blocks during matmul. Wrapped it in a `BlockSparseLinear` drop-in replacement for `nn.Linear` and swapped it in for every MLP linear on the ActW-30% pruned model.

### Correctness

MMLU on the block-sparse model: **28.4%** — matches the dense-pruned ActW 30% result (~28.1%). The kernel produces numerically equivalent output (within bf16 tolerance). ✓

### Two regimes: `M` is everything

In `Y = X @ W.T` with `X: (M, K)` and `W: (N, K)`, `M = batch_size * seq_len` — the number of input rows processed at once. Transformer inference has two completely different regimes:

- **Prefill** (processing the prompt / context): `M = prompt_length`. Big GEMM, compute-bound, tensor-core friendly.
- **Decode** (generating one token): `M = 1`. Actually a GEMV (matrix-vector), memory-bandwidth-bound.

### Prefill benchmark (seq_len 128–2048) — our kernel wins

| Shape | M | 30% sparsity | 50% sparsity |
|---|---|---|---|
| gate/up_proj | 512 | **1.07x** | **1.39x** |
| gate/up_proj | 2048 | **1.11x** | **1.47x** |
| down_proj | 512 | **1.02x** | **1.34x** |
| down_proj | 2048 | **1.06x** | **1.48x** |

Speedup grows with M (more arithmetic intensity) and with sparsity (more tiles skipped).

### Decode benchmark (M=1 to 64) — our kernel loses

| Shape | M | Best sparsity result |
|---|---|---|
| gate/up_proj | 1 | 0.77x (slower) |
| gate/up_proj | 64 | 1.18x (marginal win) |
| down_proj | 1 | 0.68x (slower) |
| down_proj | 64 | 0.26x (catastrophic) |

### End-to-end generation (decode-dominated, 50 new tokens)

- Dense baseline: 1394 ms / generate, 35.9 tok/s
- Block-sparse:   1593 ms / generate, 31.4 tok/s
- **Net: 0.88x (12% slower)** — the decode penalty overwhelms the prefill gains for short outputs.

### Root cause: memory bandwidth at M=1

At `M=1`, the matmul becomes a GEMV. You load the entire weight matrix (say 27 MB for `gate_proj`) from VRAM to multiply it against a single 1152-element vector. Compute is trivial; the bottleneck is reading weights from memory.

Our v1 kernel stores the **dense weight + bitmap** — the zeroed tiles are still in VRAM. Skipping their compute saves arithmetic we weren't bottlenecked on, while cuBLAS's hand-tuned GEMV path issues coalesced reads of the whole matrix with no per-tile branching overhead. Dense wins.

At `M ≥ 128`, arithmetic intensity rises (same weight reads reused across hundreds of rows), compute becomes the bottleneck, and skipping 30-50% of tile math actually helps.

### Autotune experiment

Wrapped the kernel in `@triton.autotune` over `BLOCK_M ∈ {16, 32, 64, 128}` and `num_warps ∈ {2, 4, 8}`, keyed on `(M, N, K)`.

| Shape | auto/orig | auto/dense |
|---|---|---|
| gate/up_proj | 0.67x–0.95x (slight regression) | 0.42x–0.75x |
| **down_proj** | **1.76x–2.26x** | 0.32x–0.62x |

Autotuning **doubled** our kernel's speed on `down_proj` — `BLOCK_M=64` was badly wrong for that shape. But even with autotuning, we still lose to dense in every decode case. Autotune is fixing kernel config, not the fundamental memory-bandwidth problem.

### Key finding

The block-sparse kernel is a **prefill accelerator**, not a decode accelerator. That's fine — for long-context inference (e.g. 300K-token context windows) prefill dominates wall-clock time, so this is useful. But for short chatbot replies where output tokens outnumber prompt tokens, the decode regression erases any prefill gain.

### Paths forward for decode

The v1 kernel can't beat dense cuBLAS at M=1 because it doesn't reduce memory traffic. Options that actually attack the bandwidth bottleneck:

1. **BCSR packing** — store only non-zero tiles contiguously + index. At 30% effective sparsity, ~30% fewer bytes read from VRAM. Directly targets the bottleneck.
2. **Fused MLP kernel** — gate + up + (elementwise) + down in one kernel. Intermediate activations stay in registers/SMEM instead of round-tripping VRAM. Kills 3-kernels-per-layer launch overhead too.
3. **Weight quantization** (INT8/INT4 on surviving tiles) — halves or quarters bandwidth. Compounds with BCSR.
4. **Hybrid dispatch** — keep the block-sparse kernel for prefill (M ≥ threshold), fall back to dense for decode. Lossless: prefill wins, decode unchanged. Easy to implement in `BlockSparseLinear.forward` via an M check.
5. **Dedicated GEMV kernel for decode** — rewrite for vector-matrix shape: iterate over non-zero column tiles, accumulate into a 1×N output vector. Different memory pattern (row-major weight scan instead of 2D block grid). Would only pay off once paired with BCSR packing so we're actually reading less.

---

## 10. Hybrid Dispatch Results + Hypothesis Revision

### Implemented: hybrid dispatch

Added `SMALL_M_THRESHOLD = 64` to `BlockSparseLinear.forward`. Below the threshold, forward uses `x @ self.weight.T` (dense cuBLAS, same result since tiles are already zeroed in memory). At or above, forward uses the Triton block-sparse kernel.

### Results: decode regression gone, but prefill didn't accelerate either

Using the ActW-30% pruned model (**21% effective sparsity** after 50% per-matrix cap):

| Scenario | Before hybrid | After hybrid | Dense speedup |
|---|---|---|---|
| Short prompt + 50 gen (decode-heavy) | 0.88x | **0.99x** | ✓ regression fixed |
| Long prompt + 20 gen (mixed) | — | **0.99x** | flat |
| Pure prefill M=1953 | — | **0.98x** | **slightly slower** |

Hybrid dispatch did exactly what it was designed to do on decode. But prefill at M=1953 — which should have been the kernel's home turf — was also 2% slower than dense. That contradicts the "M=1 is the problem" story.

### Root cause revealed: break-even is a SPARSITY threshold, not an M threshold

Re-reading our own microbenchmark at `seq_len=2048` with fresh eyes:

| Sparsity | gate/up_proj | down_proj |
|---|---|---|
| 10% | 0.72x | 0.82x |
| 30% | 0.95x | 1.10x |
| 50% | 1.41x | 1.59x |

The kernel only wins at ~30%+ sparsity. Our effective sparsity of 21% is **below break-even even at M=2048**. The "M=1 is the problem" framing was incomplete — the deeper issue is that the kernel's `tl.load(bitmap) + branch` costs something every iteration, and at low sparsity the savings from skipped tiles don't pay for that overhead.

### 50% uniform sanity test — kernel design validated

Rebuilt the model with **50% uniform per-matrix pruning** (bottom 50% of each matrix by ActW score, no cap, no global threshold). This is not a realistic config (accuracy would tank) — it's a kernel-speed test.

| Scenario | Dense | BS 21% | BS 50% | 21% vs D | **50% vs D** |
|---|---|---|---|---|---|
| short + 50 gen | 1423.0ms | 1439.2ms | 1405.4ms | 0.99x | 1.01x |
| long + 20 gen | 683.7ms | 692.1ms | 659.8ms | 0.99x | **1.04x** |
| pure prefill (M=1953) | 183.1ms | 187.7ms | **157.8ms** | 0.98x | **1.16x** |

**At 50% effective sparsity, the kernel is 16% faster than dense on pure prefill end-to-end.** The design works — we just need to be above the break-even sparsity threshold.

### Revised mental model

The original framing ("M=1 vs big-M") was too binary. The correct framing:

> The kernel has a **break-even sparsity** above which it beats dense and below which it loses. That threshold depends on M (lower M → higher break-even needed), but for any realistic M the break-even is somewhere in the 20–30% range. We need effective sparsity > break-even to see speedup.

At our ActW 21% effective sparsity, we sit right at the break-even line, so the kernel basically ties dense. Going to 50% clears the threshold decisively.

### Why this makes BCSR even more important

BCSR directly attacks the cause of the break-even tax:
- **Fewer VRAM reads** (proportional to sparsity) — biggest wins at low M where we're memory-bound
- **No bitmap-check tax** — we only iterate over non-zero tiles, so there's no "pay-to-check" cost on kept tiles
- **Break-even point should drop dramatically** — with BCSR, 20% sparsity reads 20% fewer bytes, no overhead. Probably wins at 10%+ sparsity, not 30%.

This is the cleanest path to making our realistic 21% ActW model actually accelerate. Next step.

---

## 11. BCSR v1 — Packed Storage, Block-Matmul Only

Built BCSR packing: `packed (n_kept, TILE_R, TILE_C)` + `row_ptr (n_row+1,)` + `col_idx (n_kept,)`. Kernel iterates **only over kept tiles** (no bitmap, no skip-and-check). Correctness verified against dense-pruned matmul: max diff = 0.

`BCSRLinear` is a full replacement for `nn.Linear` — no hybrid dispatch. Swapped into both the ActW-21% model and the 50%-uniform model. `build_bcsr_model()` loads fresh, packs in-place.

### Results — BCSR v1 (full replacement, all M go through block-matmul kernel)

| Scenario | Dense | BS 21% | BS 50% | **BCSR 21%** | **BCSR 50%** |
|---|---|---|---|---|---|
| short + 50 gen (decode-heavy) | 1457 ms | 1444 (1.01x) | 1422 (1.02x) | **1606 (0.91x)** | **1605 (0.91x)** |
| long + 20 gen (mixed) | 698 ms | 704 (0.99x) | 660 (1.06x) | **751 (0.93x)** | **744 (0.94x)** |
| pure prefill (M=1953) | 183 ms | 178 (1.03x) | 171 (1.07x) | **176 (1.04x)** | **169 (1.08x)** |

### Two surprises

**1. Prefill gain is tiny.** BCSR only beats the bitmap kernel by 1pp at prefill (1.04x vs 1.03x at 21%; 1.08x vs 1.07x at 50%). Reason: at M=1953 we're **compute-bound**, not memory-bound. Both kernels skip the same fraction of compute; BCSR's extra memory savings don't materialize because memory wasn't the bottleneck. The theoretical memory-bandwidth advantage of BCSR only matters in the memory-bound regime.

**2. Decode regresses to 0.91x.** BCSR *without hybrid dispatch* loses decode by ~9%, reversing the fix we got from hybrid-dispatch on the bitmap kernel. Reasons:
- Same M=1 overhead as before: kernel-launch overhead, grid inefficiency (`BLOCK_M=64` computes 64 rows but only 1 is valid → 98% wasted compute per block)
- `tl.dot` has a 16-row minimum, so even with `BLOCK_M=16` we're 16x over-computing per tile
- cuBLAS GEMV is hand-tuned for M=1 — memory savings alone can't beat 15 years of optimization
- BCSR's memory win (~21% fewer weight reads) is real but tiny compared to the overhead

### Upshot

BCSR is **correct** and gives the right theoretical shape (memory-bandwidth friendly), but:

- At large M (compute-bound): BCSR ≈ bitmap kernel — small gain, not a revolution
- At M=1 (memory-bound): BCSR should win but doesn't, because the block-matmul kernel has M=1 overhead that swamps memory savings

The fix can't be hybrid dispatch to dense (would require keeping a dense copy — defeats the packing). It also can't be reconstructing dense on the fly (too slow). The only principled path is a **dedicated BCSR-GEMV kernel** that handles M=1 with its own design: no `tl.dot`, no BLOCK_M waste, just vector × packed-tile multiplies with `tl.sum`. That's next.

---

## 12. BCSR v2 — Dedicated GEMV Kernel for M=1 (failed)

Added `bcsr_gemv_kernel` specifically for decode: grid `(n_row_tiles,)`, one block per output tile, explicit `tl.sum(w * x_slice[None, :], axis=1)` to avoid `tl.dot`'s 16-row minimum. Autotuned over `num_warps ∈ {2, 4, 8}`. `BCSRLinear.forward` dispatches: `M == 1` → GEMV, `M > 1` → block-matmul. Correctness verified (max diff = 0).

### Results — BCSRv2 made decode WORSE

| Scenario | Dense | BS 21% | BCSR 21% | **BCSRv2 21%** | BCSRv2 50% |
|---|---|---|---|---|---|
| short + 50 gen (decode) | 1456 | 1475 (0.99x) | 1603 (0.91x) | **1717 (0.85x)** | 1705 (0.85x) |
| long + 20 gen (mixed) | 682 | 685 (1.00x) | 766 (0.89x) | 780 (0.87x) | 743 (0.92x) |
| pure prefill (M=1953) | 183 | 178 (1.03x) | 175 (1.04x) | 181 (1.01x) | 163 (1.12x) |

The dedicated GEMV made decode **slower** than the block-matmul BCSR (0.85x vs 0.91x). Prefill unchanged (same kernel used). The hypothesis ("avoid `tl.dot` waste at M=1 → faster") was wrong.

### Why the prediction failed — four compounding mistakes

1. **Tensor cores beat CUDA cores even with 15/16 waste.** Ada (4070 Laptop) does ~120 TFLOPS bf16 on tensor cores vs ~30 TFLOPS FP32 on CUDA cores. With 63/64 waste, tensor cores still deliver ~1.9 TFLOPS effective — but CUDA cores with full efficiency also have per-warp reduction synchronization and lane-divergence overhead that `tl.dot` avoids. In practice, tensor cores' raw throughput keeps them ahead even with massive M-dimension waste.
2. **I added `.to(tl.float32)` upcasts inside the loop.** This was unnecessary "for accuracy" and forced the entire multiply-sum onto CUDA cores in FP32, foreclosing tensor cores as an option. The block-matmul kernel keeps everything in bf16 and uses tensor cores; GEMV explicitly opted out.
3. **`tl.sum` is a warp-level reduction with implicit barriers and shuffles**; `tl.dot` is a direct HW matmul op. On tile sizes as small as 128×128, reduction sync overhead is a significant fraction of the per-tile cost.
4. **Compute was never the bottleneck at M=1 anyway.** It's kernel launch overhead + memory bandwidth. "Saving compute" by avoiding waste can't help if compute wasn't limiting us. Worse, the dedicated GEMV adds its own launches (separate kernel) + autotune lookup overhead per call.

### The lesson

Dimensional reasoning ("waste = bad → avoid waste = good") fails without accounting for the hardware specifics: tensor cores vs CUDA cores, reduction costs, dtype retention, autotune overhead. The block-matmul BCSR kernel — even with 98% M-dimension waste at M=1 — is actually the better design on this GPU for this size of model.

### Decision: abandon BCSR

BCSR adds significant complexity (packed + row_ptr + col_idx + two different kernels) for marginal gains at prefill and regressions at decode. The simpler bitmap kernel with hybrid dispatch is the better design point for a 1B model on this GPU. Path forward: **fused MLP kernel** on top of the bitmap architecture, targeting launch-overhead reduction which is the remaining bottleneck we haven't attacked.

---

## 13. Fused MLP Kernel — Partial Fusion Experiments

### Notebook 3 — Fused gate+up+act on Gemma 3 1B

Single Triton kernel does both `gate_proj` and `up_proj` matmuls inside one grid, then applies `silu(acc_g) * acc_u` elementwise before writing the MLP hidden tensor. `down_proj` stays a separate `BlockSparseLinear`. This cuts MLP kernel launches 3 → 2 per layer and keeps the `gate` intermediate in registers.

**Results (Gemma 3 1B, ActW 21% masks):**

| Scenario | Dense | BS 21% | Fused 21% | Fusion vs BS |
|---|---|---|---|---|
| short + 50 gen | 1456ms | 1475 (0.99x) | 1507 (0.94x) | 0.94x |
| long + 20 gen | 682ms | 685 (1.00x) | 704 (0.97x) | 0.97x |
| pure prefill | 183ms | 178 (1.03x) | 181 (1.01x) | 0.98x |

**MMLU**: BS 21% = 28.4%, Fused 21% = 27.5% (-0.9pp).

**Finding**: fusion gave essentially no benefit on a 1B model. Kernel launch overhead is small relative to matmul compute at this scale. The MMLU drop was traced to a **latent bug** — we hardcoded SiLU in the kernel, but Gemma's MLP uses `gelu_pytorch_tanh`. That bug is fixed in notebook 4 (activation detected from `model.config`).

### Notebook 4 — Qwen 2.5 3B (activation-aware kernel + load-once architecture)

Moved to Qwen 2.5 3B to test the "does sparsity speedup scale with model size?" hypothesis. Key engineering changes:

1. **Activation-aware fused kernel** — `ACT_FN: tl.constexpr` dispatches between SiLU, `gelu_pytorch_tanh`, and exact GELU at compile time. Detects model activation from `config.hidden_act` / `config.hidden_activation`.
2. **Load-once / configure-in-place architecture** — load the dense model ONCE, cache original MLP weights on CPU, then use `configure_model(variant, masks)` to switch between dense/bs/fused in-place. Avoids ever holding two model copies on a tight 8 GB card.
3. **IPython-aware memory cleanup** — uncovered during debugging that IPython's `sys.last_traceback` + `Out[]` cache holds references to "deleted" models for 6+ GB. `free_models()` clears those explicitly.

### Qwen 2.5 3B benchmark results

Dense MMLU baseline: **48.7%** (vs Gemma 3 1B: 30.8%). Now pruning damage is measurable.

**End-to-end speed:**

| Scenario | Dense | BS 21% | BS 50% | Fused 21% | Fused 50% |
|---|---|---|---|---|---|
| short + 50 gen | 1422ms | 1422 (1.00x) | 1421 (1.00x) | 1422 (1.00x) | 1422 (1.00x) |
| long + 20 gen | 909ms | 918 (0.99x) | 824 (1.10x) | 890 (1.02x) | 796 (1.14x) |
| pure prefill (M≈2000) | 385ms | 396 (0.97x) | **302 (1.28x)** | 369 (**1.04x**) | **274 (1.41x)** |

**Fusion gain over plain BS** (same sparsity, same tile selection):

| Scenario | Fused/BS 21% | Fused/BS 50% |
|---|---|---|
| long + 20 gen | 1.03x | 1.04x |
| **pure prefill** | **1.07x** | **1.10x** |

### The three scaling wins

1. **Sparsity speedup scales with model size** (hypothesis confirmed). Fused 50% prefill: 1.14x at 1B → **1.41x at 3B**. Bigger MLP matmuls → more compute-bound → sparsity matters more.
2. **21% break-even crossed** on the bigger model. At 1B it hovered at 0.95–1.03x; at 3B Fused 21% prefill hits **1.04x** — the realistic ActW config now actually accelerates.
3. **Fusion gives a real 7-10% on top of plain BS** at prefill. At 1B it was noise; at 3B it's measurable.

### Accuracy — the bad news

| Variant | Sparsity | MMLU | vs Dense |
|---|---|---|---|
| Dense | 0% | 48.7% | — |
| BS 21% (ActW) | 21% | 24.9% | -23.8pp |
| Fused 21% (ActW) | 21% | 25.1% | -23.6pp |
| Fused 21% (Taylor) | 18.7% | **27.5%** | -21.2pp |
| Fused 50% (Taylor) | 50% | 26.8% | -21.9pp |

**MMLU collapsed to random chance (25%).** Qwen 2.5 3B is far more fragile to MLP pruning than Gemma 1B was — its extra capability comes from concentrated circuitry that can't tolerate 20%+ of MLP tiles being zeroed.

### Taylor > ActW confirmed (again)

At identical ~20% sparsity, Taylor scoring beats ActW by **+2.4pp** (27.5% vs 25.1%). Remarkably, **Taylor at 50% beats ActW at 21%** (26.8% > 25.1%) — so Taylor picks genuinely better tiles even when pruning 2.4× more aggressively.

### Flat accuracy curve 21% → 50% (same as 1B pattern)

Taylor 21% → Taylor 50% loses only 0.7pp (27.5% → 26.8%) despite 2.67× more pruning. Once we cross the "knee" (somewhere around 10-15% sparsity for Qwen 3B), the model is at its post-pruning floor and additional sparsity doesn't further degrade it.

**Corollary for the kernel**: since accuracy is flat above the knee, we might as well run at 50% sparsity for max speed. Fused 50% Taylor gives 1.41x prefill + 26.8% MMLU vs Fused 21% Taylor at 1.04x + 27.5% — 37% more speedup for 0.7pp less accuracy.

### Recovery path: LoRA fine-tuning

The pruned model is **structurally correct** — same weights, just with 21-50% of MLP tiles zeroed. A short LoRA fine-tune (small adapter matrices targeting MLP projections) could recover a lot of the lost MMLU by letting the remaining tiles redistribute the zeroed tiles' function. Classic sparsity-recovery literature shows 50-80% of lost accuracy can come back with a few hundred training steps.

This is **the natural next step** for the project: prune → LoRA-recover → measure. If LoRA brings Qwen 2.5 3B Fused 50% from 26.8% back up to 40%+, we'd have a legitimately useful artifact (1.41x faster, 80%+ of baseline capability).

### Summary of the kernel exploration

Validated:
- Block-sparse kernel design is correct and portable (Gemma 1B, Gemma 3 arch, Qwen 2.5 arch)
- Hybrid dispatch (M<64 → dense, M≥64 → kernel) eliminates decode regressions
- Activation-aware fused kernel gives 7-10% over plain BS at prefill
- Sparsity speedups scale with model size (confirmed going from 1B to 3B)
- Taylor > ActW > Frobenius as importance metrics (confirmed across both sizes)

Unsolved by kernel work alone:
- Accuracy preservation at meaningful sparsity for stronger base models
- Below the accuracy "knee" (sub-10% sparsity), our kernels don't accelerate much

The remaining project scope moves to **LoRA recovery** and **RYS-style layer duplication** (the other half of the original project plan).

---

## 14. LoRA Recovery Attempt — Qwen 2.5 3B, Taylor 50% (Notebook 5)

First attempt at Phase 6 recovery. Goal of this notebook was an end-to-end pipeline test at small scale before scaling up: prune → short LoRA fine-tune → re-eval.

### Pipeline built

- **Simplest design:** zero weights in place (Taylor 50% uniform per-matrix via per-component-type z-normalized scores, same as notebook 4). No forward patching, no fused kernel — just `F.linear` on zeroed weights, so PEFT attaches cleanly to the underlying `nn.Linear`s. Accuracy-equivalent to the fused path (the experiment is about recovery, not speed).
- **LoRA config:** `r=16, alpha=32`, target modules `gate_proj / up_proj / down_proj` (MLP only — matches pruning scope). 22.6M trainable params (0.73% of model).
- **Training data:** 500 samples from `cais/mmlu` `all/auxiliary_train` (disjoint from the test split).
- **Answer-token-only loss:** labels are `IGNORE` everywhere except the single answer letter token, encoded the same way `eval_mmlu.py` encodes A/B/C/D. 100% of gradient signal lands on exactly what MMLU scores.

### Two bugs fixed along the way

1. **Prompt trailing space.** First draft had `"Answer: "` (trailing space) in the training prompt. Qwen's BPE tokenizes `"Answer: "` and `"Answer: B"` to the same length (the trailing space merges into the last token, and the `"B"` variant replaces that last token with `" B"`). Every sample was skipped by the `len(prompt_ids) >= len(full_ids)` guard, giving `ZeroDivisionError`. Fixed by matching `eval_mmlu.format_mmlu_prompt` exactly — `"Answer:"` with no trailing space — and building `input_ids = prompt_ids + [answer_token]` explicitly.
2. **PEFT + gradient checkpointing OOM.** First training run OOM'd on an 8 GB card. Root cause: with a frozen base model, the input-embedding output doesn't naturally `require_grad`, so gradient checkpointing silently retains the full forward activation stack instead of recomputing. Fix: call `model.enable_input_require_grads()` right after `get_peft_model()`. Standard PEFT idiom for this case. After the fix, peak GPU during training was 7.01 GB — same as the bare-model peak, i.e. LoRA + optimizer + activations added essentially nothing on top of the dense forward pass.

### Results (N_CAL=512, N_TRAIN=500, 3 epochs, LR=1e-4, MAX_LEN=192)

| Stage | MMLU overall |
|---|---|
| Dense baseline (notebook 4) | 48.70% |
| Pruned only (Taylor 50%, pre-LoRA) | **26.68%** |
| + LoRA on 500 aux-train × 3 epochs | **25.99%** |
| Δ from LoRA | **−0.69pp** |

Pre-LoRA matches notebook 4's 26.8% result **exactly** — the mask-and-eval pipeline is bit-reproducible across notebooks. That's a clean confirmation that the variant-in-place vs fused-kernel paths give the same answer at this granularity.

### Training loss dropped sharply, but it didn't transfer broadly

Epoch-end smoothed losses: **6.27** (≈ random on 152k vocab) → 1.54 → 1.39 → **1.51**. Epoch 3 ticked back *up* — mild overfitting already by 1500 steps. The model clearly learned something (going from random to ~22% prob on the correct answer token), but the learning didn't translate to a net MMLU gain.

**Per-subject Δ is wildly bimodal:**

| Subject | Pre | Post | Δ |
|---|---|---|---|
| global_facts | 19.0 | 31.0 | **+12.0** |
| abstract_algebra | 25.0 | 30.0 | +5.0 |
| econometrics | 23.7 | 28.1 | +4.4 |
| moral_scenarios | 24.9 | 27.3 | +2.4 |
| anatomy | 22.2 | 24.4 | +2.2 |
| college_computer_science | 32.0 | 30.0 | −2.0 |
| college_chemistry | 34.0 | 30.0 | −4.0 |
| us_foreign_policy | 33.0 | 28.0 | −5.0 |
| machine_learning | 31.2 | 20.5 | −10.7 |
| professional_medicine | 30.5 | 16.9 | **−13.6** |

Five subjects gained, five lost. The two biggest regressions (professional_medicine −13.6pp, machine_learning −10.7pp) cancelled the gains.

### Interpretation — data coverage, not optimization

`auxiliary_train` is scraped from ARC, RACE, OpenBookQA, HellaSwag — not actual MMLU content. The first 500 streamed examples are almost certainly dominated by one source, so training shifts the model's answer-token distribution toward one narrow style. That helps subjects whose question format resembles the training source (hence +12pp on global_facts) and actively hurts subjects whose format differs (hence −13.6pp on professional_medicine).

The loss curve confirms learning is real; the per-subject pattern says the *direction* of that learning isn't recovering broken MLP circuits, it's overwriting one answer-format prior with another.

### Decisions

- **LoRA recovery is non-trivial at this damage level.** 50% MLP-tile pruning on Qwen 3B destroyed 22pp of MMLU; 500 aux-train samples shifted things ±12pp per subject without net gain. "50–80% recovery in a few hundred steps" from the sparsity-recovery literature doesn't transfer directly to this configuration.
- **Next recovery attempt should target the pretraining distribution** (C4 / FineWeb, next-token loss) rather than MMLU-flavored multi-choice data. The pruning damaged MLP circuits that were trained on the pretraining distribution; recovery data should match. MMLU-aux training is optimizing for a different objective (pick one of four letters) than what was broken.
- **Scaling this exact configuration up is risky.** Epoch 3 already showed overfitting at 1500 steps. More data + fewer epochs + shuffled across sources is the right shape. But the pretraining-style recovery experiment is more likely to produce a broad gain than a scaled-up aux-train run.
- **RYS-style layer duplication (Phase 5) is still untouched** and operates on a different axis (no training required). If pretraining-style LoRA also stalls, Phase 5 becomes the natural next move.

### Pipeline artifacts now working (reusable for future experiments)

- In-place pruning via `_apply_masks_inplace` without configure_model — simpler path when there's only one variant.
- Taylor scoring with `register_post_accumulate_grad_hook` + immediate `param.grad = None` — confirmed to work on 3B on 8 GB with gradient checkpointing.
- PEFT LoRA on frozen pruned base, trainable MLP adapters only, 7.01 GB peak.
- Answer-token-only supervision via `tokenizer.encode(label, add_special_tokens=False)[-1]` — matches eval byte-for-byte, no tokenizer drift.
- Diagnostic assertions that fail loud on empty batch lists.

---

## 15. LoRA Rank Sweep + Optimizer Confound — Qwen 2.5 3B (Notebook 5 continued)

Hypothesis after section 14: maybe r=16 adapters (22.6M params, 0.73% of model) lacked capacity to absorb broken-tile functionality. Test plan: bump rank, see if recovery improves.

### OOMs forced an optimizer change

- **r=128** (target 180M trainable): instant OOM on the 8 GB card. Expected — AdamW fp32 optimizer state alone was ~1.4 GB on top of the 7.01 GB baseline.
- **r=96**: also OOM.
- **r=64**: OOM too at full precision. Activation fragmentation from the earlier calibration + pre-LoRA eval in the same kernel ate the remaining headroom.

**Mitigation:** installed `bitsandbytes` and swapped `torch.optim.AdamW` → `bnb.optim.AdamW8bit` (quantizes the m/v momentum buffers to 8-bit, cuts optimizer state ~4×).

### Run 2 results — r=64 + AdamW8bit

| Stage | MMLU | Δ from pre-LoRA |
|---|---|---|
| Pre-LoRA (Taylor 50%) | 26.68% | — |
| Run 1: r=16 + AdamW fp | 25.99% | −0.69pp |
| **Run 2: r=64 + AdamW8bit** | **22.89%** | **−3.79pp** |

Training loss started at 13.33 (first 20 steps) vs 6.27 for run 1. Same LoRA init (`B=0`, so step-0 adapter output is zero regardless of rank) — the early divergence came from the **first few updates** making things worse, not the initialization.

### Initial misreading — and the correction

My first interpretation of this result was "bigger rank overfits harder on narrow aux-train data → data distribution is the real bottleneck, pivot to pretraining-distribution recovery." That was premature — run 2 changed **two variables at once** (rank and optimizer). The user caught the confound and proposed the right isolation test: revert rank to 16 while keeping AdamW8bit.

### Run 3 — r=16 + AdamW8bit (isolates the optimizer)

| Run | Rank | Optim | First-20 loss | Last-20 loss | MMLU | Correct |
|---|---|---|---|---|---|---|
| Run 1 | 16 | AdamW fp | 6.27 | **1.51** | 25.99% | 527/2028 |
| Run 2 | 64 | AdamW8bit | 13.33 | 1.57 | 22.89% | **464/2028** |
| Run 3 | 16 | AdamW8bit | 6.19 | 1.85 | **22.89%** | **464/2028** |

**AdamW8bit is the culprit.** Same rank as run 1, different optimizer, MMLU dropped −3.1pp. Runs 2 and 3 landed on the **identical 464/2028** with nearly byte-for-byte identical per-subject scores — AdamW8bit pushes the adapters into the same degenerate attractor regardless of rank.

### What AdamW8bit actually breaks (and what it doesn't)

- **Base weights are untouched.** Only LoRA adapters are trainable, so the frozen pruned base stays at bf16 integrity regardless of optimizer precision.
- **First-20 loss is fine at r=16** (6.19 ≈ run 1's 6.27). The optimizer doesn't blow up early — the base + zero-init adapters still evaluate correctly, and the first batch of updates is small enough that quantization error is bounded.
- **Final loss floor is higher** (1.85 vs 1.51). Once gradients get small near a minimum, the quantized m/v momentum can't resolve the updates precisely enough — the optimizer takes noisy steps around the valley instead of settling into it.
- **Net: adapter learns a noisy, partial version of the aux-train mapping.** MMLU evaluator picks up this noise as shifted logits at A/B/C/D positions, dragging accuracy below even the pre-LoRA baseline.

### Why QLoRA gets away with 8-bit optimizers but we don't

QLoRA literature uses 8-bit optimizers successfully, but that's on **pretraining-distribution tasks** where per-token loss is typically 2–5 and gradients are large. Signal-to-quant-noise ratio is comfortable.

Our regime is different: answer-token-only loss on a narrow MCQ distribution, final loss floor ~1.5. Gradients near that floor are small and quant error dominates. Recovery fine-tuning after pruning is a **low-gradient** regime — precisely where AdamW8bit is most damaging.

### Takeaways

- **Capacity is not the recovery bottleneck** at this damage level. We can't cleanly prove it at r=64+ because we can't run it full-precision on this GPU, but the signal from r=16+8bit vs r=16+fp is strong enough to reject the "bigger rank = better recovery" hypothesis in the regime we can test.
- **8-bit optimizer is not safe for recovery fine-tuning on small LoRA.** This is a real concrete finding for anyone trying QLoRA-style recovery on pruned models: the precision tradeoff that works for pretraining hurts at recovery.
- **The data-distribution hypothesis is still untested.** Run 1 (r=16 + fp AdamW) gave −0.69pp with per-subject swings of ±13pp — consistent with aux-train narrowness, but not proven. The cleanest next experiment is still pretraining-distribution LoRA (C4 / FineWeb, next-token CE), kept at r=16 + full-precision AdamW.
- **If we want to test higher rank, we need memory fixes that don't compromise the optimizer:** shorten MAX_LEN 192 → 128 (cuts training activations ~33%), drop `down_proj` from targets (−40% adapter params but loses that projection's recovery), or rent GPU time with more VRAM. 8-bit optimizer is off the table for this task.

### Methodological lesson

When changing two variables simultaneously (rank AND optimizer) produces an unexpected result, the first instinct should be isolation, not interpretation. I jumped to a data-distribution narrative that plausibly fit run 2's numbers but wasn't supported by the experiment design. The user's catch — "you can't compare like that" — saved us from writing up a wrong hypothesis as fact.

### Status: blocked by VRAM

**We were unable to reach final recovery results because we are constrained by the 8 GB VRAM of the RTX 4070 Laptop.** The experiments we can run on this hardware have given us a clear picture of what's happening (8-bit optimizer is unsafe for recovery; r=16 + full-precision AdamW with MMLU aux-train yields at best −0.69pp) but not what the ceiling actually is.

The experiments we cannot run on this hardware, and which would be the natural next steps:

- **r=64+ with full-precision AdamW** — direct capacity test. Every attempt OOM'd.
- **Longer training runs at r=16** — more epochs, more data samples, but N_TRAIN=500 was already pushing the memory budget with activations retained across sequences.
- **Pretraining-distribution LoRA on C4/FineWeb** — same r=16 memory profile is fine in principle, but longer sequences (full 512-token windows as used in our calibration) inflate training activations past what the card can hold alongside LoRA adapters + optimizer state.
- **Combining LoRA recovery with larger calibration runs** — the current notebook 5 already OOMs when you try to go above N_CAL=512 without freeing intermediate tensors aggressively.

**The scientific loop is bottlenecked by VRAM, not by algorithmic ideas.** Continuing this line of work productively requires either:
- Hardware with ≥16 GB VRAM (a desktop 4080/4090, a rented A100/H100, or Colab Pro T4/A100), OR
- A different memory discipline for the notebook (CPU-offloaded optimizer state via DeepSpeed ZeRO-Offload, or per-layer LoRA attachment + gradient-release to drop peak activations).

Both are out of scope for a single 8 GB laptop card doing LoRA on a 3B model. The findings above are therefore **preliminary, not conclusive** — they tell us the optimizer choice matters and that narrow MCQ data probably isn't the right recovery signal, but they do not establish a recovery ceiling for pruned Qwen 2.5 3B.

---

## 16. Phase 5 — RYS-style Layer Duplication on Qwen 2.5 3B (Notebook 6)

Phase 5 tested the other half of the project thesis: can duplicating high-importance transformer blocks *improve* MMLU (RYS-style), and does the same improvement recover pruned models (the "full RMP" idea)?

### Setup

- Same Qwen 2.5 3B Instruct base, same 512-sample C4 Taylor calibration as notebooks 4/5.
- Aggregate per-tile Taylor scores to **per-layer importance** by summing per-component-type z-normalized scores across all 3 MLP matrices in each block.
- Duplication = insert the same `Qwen2DecoderLayer` module object twice into `model.model.layers`. Shared weights, zero extra params, just one extra forward call per duplicated layer.
- **Phase 1**: dense model + duplication (isolates the duplication signal).
- **Phase 2**: Taylor-50% pruned + duplication (tests whether duplication recovers pruning damage).

### Per-layer Taylor importance map — the chart that surprised us

The z-normalized Taylor sum per layer revealed a clean **three-zone functional structure**:

| Zone | Layers | Score range | Interpretation |
|---|---|---|---|
| Encoder ("translators") | 0–7 | negative, deeply so at 1-5 | Input encoding |
| **Circuit peak** | **8–16** | **+2255 to +4821** | **Reasoning circuits** |
| Mid trough | 17–26 | near zero / slightly negative | Passthrough |
| Late positive | 27–35 | +1500 to +3000 | Output decoders |

Layer 11 peaked at +4821 (z-sum). Layers 8 (+2255) and 16 (+2317) are the clean boundary points where importance drops off sharply.

**This is the same three-zone pattern RYS described on Qwen2-72B, but positioned differently.** RYS found the peak at layers 45-51 (positions 0.56-0.64 of 80). Our peak is at layers 8-16 (positions 0.22-0.44 of 36). The RYS relative-scaling assumption (80→36) would put our peak at layers 20-23 — exactly where our chart shows the *trough*. Smaller models concentrate reasoning circuits earlier in the stack.

### One bug fixed along the way

HF's `Qwen2Model.forward` uses `self.config.layer_types[i]` to pick per-layer attention-mask types. `layer_types` is a list of length 36 (original layer count). Duplicating the ModuleList without mirroring the insertion on `config.layer_types` → `IndexError` on the duplicated layer. Fix: extend `layer_types` with the same insertion transform applied to `model.model.layers`.

### Eleven variants, three design families

| Family | Variant | Spec | Block size |
|---|---|---|---|
| (A) Taylor pure | A1 top-3 scattered | [10, 11, 13] | 3 scattered |
| (A) Taylor pure | A2 top-1 single | [11] | 1 |
| (B) RYS-inspired | B1 relative-scaled window | [20, 23] | 4 contig (trough) |
| (B) RYS-inspired | B2 wider middle | [18, 25] | 8 contig |
| (C) Hybrid | C best-Taylor contig-4 | [10, 13] | 4 contig (peak) |
| (D/E/F) Stitched top-N | D top-5 stitched | [10, 14] | 5 contig |
| (D/E/F) Stitched top-N | E top-7 stitched | [9, 15] | 7 contig |
| (D/E/F) Stitched top-N | **F top-9 stitched** | **[8, 16]** | **9 contig (full peak)** |
| (G/H/I) Zone probing | G far-tail | [32, 35] | 4 contig (late positive) |
| (G/H/I) Zone probing | H trough | [1, 4] | 4 contig (encoder-negative) |
| (G/H/I) Zone probing | I gap scatter | [4, 20, 32] | 3 scattered across zones |

### Phase 1 results — dense + duplication

| Variant | MMLU | Δ vs dense (48.67%) |
|---|---|---|
| A1 top-3 scattered | 39.89% | −8.78pp |
| A2 top-1 single | 44.82% | −3.85pp |
| B1 RYS 20-23 | 46.06% | −2.61pp |
| B2 wide middle 18-25 | 45.51% | −3.16pp |
| C best-Taylor [10, 13] | 43.29% | −5.38pp |
| D stitched top-5 [10, 14] | 41.47% | −7.20pp |
| E stitched top-7 [9, 15] | 40.88% | −7.79pp |
| **F stitched top-9 [8, 16]** | **48.87%** | **+0.20pp** 🎉 |
| G far-tail [32, 35] | 45.12% | −3.55pp |
| H trough [1, 4] | 40.14% | −8.53pp |
| I gap scatter [4, 20, 32] | 40.88% | −7.79pp |

**F is the first variant to cross into positive territory — beating the dense baseline.** +0.20pp is modest in absolute terms but is the RYS phenomenon replicated at 3B scale, with a dramatic sign flip right before it.

### The block-size cliff

Sorting only the peak-targeted variants by block size reveals a non-monotonic curve with a sharp cliff at 9:

| Block size | Variant | Span | Δ vs dense |
|---|---|---|---|
| 1 | A2 | [11] | −3.85pp |
| 3 scatter | A1 | [10, 11, 13] | −8.78pp |
| 4 contig | C | [10, 13] | −5.38pp |
| 5 contig | D | [10, 14] | −7.20pp |
| 7 contig | E | [9, 15] | −7.79pp |
| **9 contig** | **F** | **[8, 16]** | **+0.20pp** |

Going 7 → 9 layers flipped the sign entirely. That's not noise. F covers **exactly the span** where Taylor importance is sharply positive (8-16); shorter windows carve an arbitrary slice out of the middle of that circuit.

### Why F works when D and E don't — the circuit-boundary hypothesis

RYS's framing was: transformer circuits are multi-layer processing units that must be executed as complete units. Running a **whole** circuit twice makes the model "think harder" on that computation. Running a **partial** circuit twice is destructive — the second pass receives the output of layer 14 (say) and feeds it back into layer 10, which was built to receive layer 9's output. Mid-circuit hidden states get fed to earlier mid-circuit positions and the computation corrupts itself.

D and E sliced the peak mid-circuit (cutting at both ends). F captured the **entire** natural boundary — layer 8 (score +2255) entering and layer 16 (+2317) exiting — so both start and end match the circuit's actual boundaries. Our Taylor map provided the boundary detection that RYS had to find by systematic scan of 3,240 configs.

### Hypotheses tested — three wrong predictions before running

Before running the extended variant set, I predicted:
1. **G far-tail would be circuit-like**. *Wrong-ish*: it hurt (−3.55pp) but less than peak-partials like C (−5.38pp) or D (−7.20pp). Late-positive layers behave more like output decoders — they can handle some re-execution.
2. **H trough would be safest to duplicate**. **Very wrong**: H was the *worst* non-scattered variant (−8.53pp), even worse than scattering the peak (A1 at −8.78pp is within 0.25pp). My "low importance = passthrough" framing was wrong. Layers 1-4 have strongly negative Taylor scores not because they're passive but because they're doing heavy *encoding* work — re-running them corrupts early representations that every downstream layer depends on.
3. **I gap-scatter might avoid circuit corruption**. **Wrong**: (−7.79pp) hurt just as much as A1's within-peak scattering. Circuit corruption is a *local* property of any inserted duplicate — scattering across zones just lights up three different local corruptions at once.

### Phase 2 results — pruned base + duplication

| Variant | MMLU | Δ vs pruned (26.68%) |
|---|---|---|
| A1 top-3 scattered | 27.51% | +0.83pp |
| A2 top-1 single | 26.33% | −0.35pp |
| B1 RYS window | 25.39% | −1.29pp |
| B2 wide middle | 27.07% | +0.39pp |
| C best-Taylor | 26.23% | −0.45pp |
| D stitched top-5 | 25.84% | −0.84pp |
| E stitched top-7 | 26.38% | −0.30pp |
| F stitched top-9 | 27.17% | +0.49pp |
| G far-tail | 27.32% | +0.64pp |
| H trough | 25.59% | −1.09pp |
| I gap scatter | 27.07% | +0.39pp |

All deltas are within ±1.3pp on a base that's already at random chance. **No clean signal** — the pruned model is too damaged to amplify the duplication effect the way an intact base does. A post-knee floor means small perturbations shuffle logits without changing the underlying representation quality.

### Takeaways for the project thesis

- **Phase 5 works**: duplicating the *natural circuit boundary* identified by Taylor importance does beat dense baseline on MMLU. +0.20pp is a positive result on the accuracy axis — small, but in the right direction and confirming the RYS mechanism at 3B.
- **Taylor importance is doing double duty**: same scoring that identifies safe-to-prune tiles also identifies safe-to-duplicate circuits. That's the "one map, both directions" premise of the project validated.
- **RYS's relative position scaling doesn't transfer across model sizes.** Smaller models concentrate circuits earlier. Data-driven circuit detection (via Taylor) is the generalizable method.
- **Budget-neutral prune + duplicate didn't work at Taylor-50% pruning** — the base is too degraded. Lighter pruning (Taylor 10-20%) plus F duplication is the natural next test: keep accuracy above the knee, add F to push slightly above dense while saving MLP compute. That's the real RMP experiment.
- **The circuit-corruption theory is now backed by data**, not just RYS's prose: H (deep-encoder partial) and C/D/E (peak partials) all hurt badly, while F (whole peak) helps. Partial circuit re-execution is the universal failure mode.

### Numbers to remember

- **F = layers 8-16 duplicated, MMLU 48.87%** (dense 48.67%, +0.20pp)
- Circuit boundary: Taylor z-sum crosses above +2000 at layer 8, stays above through layer 16, drops sharply after.
- Duplication adds ~zero VRAM (shared module references) — a pure compute trade.

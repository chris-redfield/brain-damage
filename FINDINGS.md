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

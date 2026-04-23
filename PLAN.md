# Regional Magnitude Pruning (RMP)

**Accelerating LLM inference by pruning contiguous weight regions and skipping their computation at runtime — then duplicating the regions that matter most.**

Inspired by [LLM Neuroanatomy / RYS](https://dnhkng.github.io/posts/rys/) — that post showed transformers develop functional regions (some layer blocks matter, others don't). We combine both sides of that insight:

1. **Prune the dead weight** — identify and remove low-importance contiguous regions, skip their compute at runtime for real speedup.
2. **Duplicate the critical circuits** — the same importance map that tells us what to cut also tells us what matters most. Duplicate those high-importance blocks (the RYS trick) so the model "thinks harder" where it counts.

The net result: parameter count stays roughly the same as the original, but compute is *redistributed* — away from dead regions, toward the model's core reasoning circuits. Faster AND smarter.

## Why not standard pruning?

Standard magnitude pruning zeros out individual weights scattered across the matrix.

The matrix stays the same shape, and GPUs can't skip the zeroed ops — you get no speedup without specialized sparse-matrix hardware (which is limited and often slower for moderate sparsity).
 
**Our approach prunes contiguous rectangular tiles**, so a custom CUDA kernel can skip entire block-level GEMMs.

---

## Baseline Results

Model: `google/gemma-3-1b-it` (instruction-tuned, bf16, RTX 4070 Laptop 8GB)

| Metric | Accuracy |
|---|---|
| **OVERALL** | **30.8% (624/2028)** |

> The pretrained variant (`gemma-3-1b-pt`) scored 23.8% — random chance level,
> unusable for measuring pruning degradation. The instruction-tuned model gives
> a clear signal above random (25%), making pruning impact measurable.
>
> Best pruning result so far: **Gradient (Taylor) at 20% → 28.3%** (only -2.5pp
> from baseline).

---

## Phase 1: Regional Importance Scoring

**Goal:** For each weight matrix in the model, divide it into tiles and score how much each tile contributes.

### 1.1 Tile definition
- Divide each weight matrix `W [out_features, in_features]` into non-overlapping rectangular tiles of size `(B_r, B_c)`.
- Default tile size: `(128, 128)` — aligns with GPU warp/block granularity. Experiment with `(64, 64)` and `(256, 256)` as well.
- For matrices whose dimensions aren't multiples of the tile size, the boundary tiles are never pruned (always kept).

### 1.2 Importance metric
Candidate metrics to evaluate (start simple, get fancier if needed):

1. **Block magnitude (L1/L2 norm):** `score(tile) = ||W_tile||_F`. Cheapest to compute. Baseline.
2. **Activation-weighted magnitude:** `score(tile) = ||W_tile||_F * mean(||x_tile_inputs||)` over a calibration set. Accounts for the fact that a large-weight tile fed by near-zero activations doesn't matter.
3. **Reconstruction error:** For each tile, zero it out and measure the L2 change in the layer's output over the calibration set. Most accurate but O(n_tiles) forward passes per layer — expensive. Use as ground truth to validate cheaper metrics.

### 1.3 Calibration dataset
- Use a small, diverse calibration set (~512-1024 samples) from a mix of tasks (e.g., subsets of C4, GSM8K, MMLU).
- Run a single forward pass over calibration data to collect input activations per layer for the activation-weighted metric.

---

## Phase 2: Pruning Strategy

**Goal:** Given tile importance scores, decide which tiles to zero out.

### 2.1 Global vs. per-layer budgets
- **Per-layer uniform:** Prune the bottom X% of tiles in every layer. Simple but ignores that some layers are more important (the RYS insight).
- **Global ranking:** Pool all tiles across all layers, rank by importance, prune the bottom X% globally. Naturally prunes more from less-important layers.
- **Layer-sensitive:** Use the RYS heatmap idea — run a quick sensitivity scan per layer (e.g., zero out 50% of each layer independently, measure perplexity delta). Assign per-layer pruning budgets inversely proportional to sensitivity. **This is the most promising approach.**

### 2.2 Pruning schedule
- Start with aggressive pruning targets: 10%, 20%, 30%, 50% of tiles removed.
- For each target, measure (a) model quality and (b) inference speedup.
- Optionally: gradual pruning with light fine-tuning between steps (but the v1 goal is zero training, like RYS).

### 2.3 Which layers to target
- **MLP layers:** gate_proj, up_proj, down_proj. These are the bulk of parameters and compute. **Primary and only target for now.**
- **Attention projections:** Q, K, V, O matrices — **excluded from pruning** for now. They are tightly coupled (Q/K/V must stay aligned for each head), and tile-pruning risks breaking head coherence in unpredictable ways. Left as a future experiment.
- **Embeddings / LM head:** Leave untouched (they handle I/O encoding, the "translator" layers from the RYS post).

---

## Phase 3: Custom Sparse Operator

**Goal:** A CUDA kernel (or Triton kernel) that exploits the block-sparse pattern to skip zeroed tiles.

### 3.1 Sparse format
- Store a **block-sparse bitmap** per weight matrix: a boolean grid of shape `(out_features // B_r, in_features // B_c)` indicating which tiles are non-zero.
- Store only the non-zero tiles in a packed contiguous buffer + an index mapping tile positions.
- This is essentially Block Compressed Sparse Row (BCSR) format at the tile level.

### 3.2 Kernel design
- **Triton first** (easier to iterate), CUDA later if needed for performance.
- The kernel:
  1. Reads the bitmap for the current weight matrix.
  2. For each output tile row, iterates only over non-zero input tile columns.
  3. Performs the tile-level GEMM (B_r x B_c) and accumulates.
- Key optimization: tile size must match Triton's `BLOCK_SIZE` for coalesced memory access.

### 3.3 Integration
- Replace `nn.Linear` modules in the model with a custom `BlockSparseLinear` that:
  - Stores weights in packed BCSR format.
  - Calls the Triton kernel for forward pass.
  - Provides a `from_dense(linear, mask)` classmethod for easy conversion.
- The model loads normally (dense), we run the pruning pass, then convert in-place.

---

## Phase 4: Evaluation & Tradeoff Analysis

**Goal:** Quantify the speed vs. quality tradeoff rigorously.

### 4.1 Quality metrics
- **Perplexity** on held-out text (WikiText-2, C4 validation).
- **Benchmark scores:** MMLU, GSM8K, ARC-Challenge, HellaSwag — a compact but diverse suite.
- **Qualitative:** Spot-check generations for coherence, check for the "brain damage" symptoms described in the RYS post (stuttering, loops, personality shifts).

### 4.2 Speed metrics
- **Latency:** Time per token (prefill and decode separately) at batch_size=1.
- **Throughput:** Tokens/sec at various batch sizes.
- **Memory:** Peak VRAM usage (the packed format should also save memory).
- Compare against: (a) the dense baseline, (b) standard unstructured pruning at same sparsity, (c) quantization (GPTQ/AWQ) at similar compression ratio.

### 4.3 Tradeoff curves
- Plot quality (y-axis) vs. speedup (x-axis) at different pruning percentages.
- The sweet spot is where the curve has a "knee" — max speedup before quality degrades sharply.
- Produce per-layer heatmaps showing which layers/regions got pruned — compare with RYS importance maps.

---

## Phase 5: Block Duplication (the RYS flip side)

**Goal:** Use the same importance heatmap to identify the highest-scoring blocks/layers and duplicate them.

- The importance scoring from Phase 1 produces a full map: low-importance tiles get pruned (Phase 2), but the **high-importance tiles** are candidates for duplication.
- At the layer level: duplicate entire high-importance transformer blocks (like RYS did with layers 45-52 in Qwen2-72B).
- At the tile level: duplicate high-importance tiles within a layer — effectively giving more capacity to the sub-circuits that matter.
- **Budget-neutral duplication:** prune X% of tiles, then duplicate the top-scoring tiles to fill roughly the same parameter budget. The model stays the same size but redistributes capacity from dead regions to critical ones.
- Measure quality gains from duplication independently, then combine with pruning for the full picture.

---

## Phase 6: Stretch Goals

- **Whole-layer removal (GMP at layer granularity):** Use the same importance-scoring framework but at layer level — score entire transformer blocks by their contribution to output quality, and drop the weakest ones entirely. Much bigger speedups than tile pruning (skip a full block of attention + MLP), but coarser. Natural progression: use tile-level heatmaps to identify nearly-dead layers, then remove them. This is the RYS insight in reverse — they found which layers to duplicate, we find which to delete.
- **Pruning + quantization:** Apply INT4 quantization to the surviving tiles for compounding speedups.
- **Fine-tuning recovery:** After pruning, do a short fine-tune (LoRA on the remaining weights) to recover lost quality. Measure how much of the gap closes.
- **Dynamic sparsity:** Different prompts activate different regions. Could we have a lightweight router that selects which tiles to use per-input?

---

## Implementation Order

```
Step 0: Setup & model loading
        - Pick a target model (Qwen2-7B or Llama-3-8B — small enough to iterate fast)
        - Load with HuggingFace transformers, verify baseline benchmarks

Step 1: Tile importance scoring (Phase 1)
        - Implement tile partitioning
        - Implement L1/L2 block magnitude scoring
        - Implement activation-weighted scoring with calibration pass
        - Visualize importance heatmaps per layer

Step 2: Pruning (Phase 2)
        - Implement global and per-layer pruning strategies
        - Apply pruning at 10/20/30/50% targets
        - Measure perplexity after each

Step 3: Custom kernel (Phase 3)
        - Implement BCSR packing format
        - Write Triton block-sparse matmul kernel
        - Implement BlockSparseLinear module
        - Benchmark kernel speed vs. dense torch.mm

Step 4: End-to-end evaluation (Phase 4)
        - Full benchmark suite at each sparsity level
        - Generate tradeoff curves
        - Write up findings

Step 5: Block duplication (Phase 5)
        - Use importance heatmap to identify top-scoring blocks/layers
        - Implement budget-neutral duplication (prune bottom, duplicate top)
        - Measure quality vs. original, pruned-only, and duplicated-only
        - Find the optimal prune/duplicate ratio

Step 6: Stretch goals (Phase 6)
        - Whole-layer removal, quantization, fine-tuning recovery, dynamic sparsity
```

---

## Tech Stack

- **Python 3.10+**
- **PyTorch 2.x**
- **HuggingFace Transformers** — model loading, tokenizer, generation
- **Triton** — custom GPU kernels
- **lm-eval-harness** — standardized benchmarking
- **wandb** (optional) — experiment tracking
- **Target hardware:** RTX 4090 / A100 (whatever is available)

---

## Open Questions

1. **Optimal tile size?** 128x128 is a guess based on GPU architecture. Needs empirical tuning — too small = overhead from sparse indexing, too large = can't prune granularly enough.
2. **Does block-sparse actually beat dense on modern GPUs?** Tensor cores are very efficient for dense ops. We need the sparsity to be high enough (>30%?) for the skipped compute to outweigh the indexing overhead. This is the make-or-break question.
3. **How does this compare to head pruning?** Removing entire attention heads is a coarser version of this idea. Our approach is more flexible (can prune sub-head regions, MLP tiles, etc.) but also more complex.
4. **Interaction with KV cache?** Pruning attention projections might allow KV cache compression too — worth investigating.

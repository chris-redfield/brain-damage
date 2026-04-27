# General Findings — Brain-Damage Project

The big ideas we learned, stated cleanly. For the chronological experiment log see `FINDINGS.md`.

---

## 1. Taylor importance beats magnitude and activation-weighted, consistently

**Finding.** Gradient-based Taylor importance (`Σ |w · ∂L/∂w|`) outperforms both Frobenius norm and activation-weighted scoring at every sparsity level we tested, on both Gemma 3 1B and Qwen 2.5 3B. At 20% sparsity on Gemma, Taylor preserves 28.3% MMLU vs ActW's 22.9% (dense 30.8%).

**Why it matters.** Magnitude-only metrics (Frobenius) can't tell a tile "used near zero activations" from a tile "doing heavy lifting." Activation-weighting fixes half of that. Taylor fixes all of it by answering the exact right question: "how much would the loss change if this tile were zero?" That's a first-principles objective, not a proxy.

**What to take away.** When picking tiles to prune, spend the extra ~5 min of forward+backward calibration for Taylor. It's strictly better unless you're memory-constrained past the point where gradients fit.

---

## 2. Per-component-type z-normalization is essential

**Finding.** Weight matrices have systematically different scales (`gate_proj` mean abs weight ≈ 0.022, `down_proj` ≈ 0.009 on Gemma). A global threshold wipes out whichever component has smaller magnitudes first, even if it's doing real work. Z-normalizing *per component type* (all `gate_proj` together, all `up_proj` together, all `down_proj` together) fixes the cross-scale imbalance while preserving the cross-layer signal about which layers have more expendable weight.

**Why it matters.** The subtle failure mode is "the math works, the model breaks." Without normalization, `down_proj` can hit 100% pruning in some layers while `gate_proj` goes largely untouched. That's catastrophic even at small global sparsity targets.

**What to take away.** Any comparison of importance scores across different weight matrices needs normalization. The specific form (per-type z-score) is simple and well-motivated.

---

## 3. The accuracy knee is a sparsity threshold, not a gradient

**Finding.** Accuracy vs sparsity is nearly flat after a knee around 10-20% sparsity. On Qwen 2.5 3B, Taylor 20% gives 26.78%, Taylor 50% gives 26.68% — almost identical. Going from 20% to 50% barely moves MMLU because we're already at the random-chance floor.

**Why it matters.** "Quality vs sparsity tradeoff" isn't a smooth curve you tune along — it's a cliff. Once you're past the knee, more aggressive pruning is free (no additional accuracy loss) but also useless (you're already broken). Once you're below the knee, every percentage point matters.

**Consequence for speedup.** Above the knee, run at whatever sparsity maximizes kernel gain. At 50% sparsity on Qwen, we got 1.41× prefill speedup with the same MMLU as 20% — 37% more speedup for no additional accuracy cost.

**Stronger models have LOWER knees.** Gemma 1B tolerates ~30% sparsity before collapsing. Qwen 3B's knee is below 10%. Harder models concentrate more capability in the same parameter count, leaving less redundancy for pruning to exploit.

---

## 4. Block-sparse kernels: break-even is a sparsity threshold, not a sequence-length threshold

**Finding.** Our first mental model for kernel speedup was "wins at big M (prefill), loses at M=1 (decode)." That's partly true but wrong as the primary axis. The real threshold is **sparsity**: at <~25% the bitmap-check-and-branch overhead in the kernel exceeds the compute saved by skipped tiles, regardless of M. At 50% sparsity the kernel wins even at moderate M.

**Why it matters.** "Below break-even sparsity" and "above break-even sparsity" are completely different regimes. The same kernel that loses by 3% at 21% sparsity wins by 41% at 50% sparsity. If you're building a block-sparse kernel, the first question isn't "what M are we at" — it's "are we sparse enough for the skips to pay for the bookkeeping."

**Hybrid dispatch is still essential.** Below some M (~64), even at good sparsity, dense cuBLAS GEMV beats any Triton block-matmul because decode is memory-bandwidth-bound and our bitmap kernel doesn't reduce memory traffic. Route small-M through dense.

---

## 5. Sparsity speedup scales with model size

**Finding.** Same 50%-sparse MLP kernel: 1.14× prefill at 1B, 1.41× at 3B. Fused version scales similarly.

**Why it matters.** Bigger MLP matmuls are more compute-bound. The fraction of compute saved by tile skips matters more when compute is the bottleneck. This is a positive scaling story — if kernel work barely pays at 1B, extrapolate to "pays more at 7B, more at 70B."

**What to take away.** Don't reject a sparse kernel because it gives 3% on a small model. Test at least one size up before concluding.

---

## 6. Tensor cores with massive waste beat CUDA cores with full efficiency

**Finding.** We tried a dedicated BCSR-GEMV kernel for the M=1 decode regime, designed to avoid `tl.dot`'s 16-row minimum waste (where most of the tensor-core rows compute unused output). Dimensional reasoning said "less waste = faster." It was actually slower (0.85× vs 0.91×).

**Why it happened.** Ada-class GPUs do ~120 TFLOPS bf16 on tensor cores vs ~30 TFLOPS fp32 on CUDA cores. Even with 15/16 rows wasted on tensor cores, that's still more effective throughput than CUDA cores at full efficiency. Plus reduction sync costs (`tl.sum` has implicit barriers), dtype upcasts that forecloses tensor-core eligibility, and autotune overhead.

**What to take away.** Hardware-specific reasoning beats dimensional reasoning on GPU kernels. "Eliminate waste" is a good instinct on CPU; on Ada tensor cores it can be the wrong optimization target.

---

## 7. 8-bit optimizers are unsafe for low-gradient recovery fine-tuning

**Finding.** Switching `torch.optim.AdamW` → `bnb.optim.AdamW8bit` at rank 16 (everything else identical) dropped MMLU by 3.1pp and landed training in the same degenerate attractor regardless of rank. The quantized `m` and `v` momentum buffers can't resolve small updates near a loss floor.

**Why QLoRA gets away with it.** Pretraining-distribution fine-tuning has per-token loss ~2-5 and large gradients throughout. Signal dominates quantization noise. Recovery after heavy pruning is a *low-gradient* regime — you're trying to fit a narrow answer distribution where gradients become tiny once the model roughly matches the targets. There, quant noise dominates signal.

**What to take away.** "QLoRA worked for X so we'll use it for Y" can fail if X and Y differ in gradient regime. Stick with full-precision optimizer state for recovery fine-tuning; use 8-bit only when you know the loss floor is high enough that signal dominates.

---

## 8a. Offline teacher distillation recovers structured-pruning damage at any sparsity tested

**Finding.** Offline knowledge distillation — caching the unpruned teacher's top-K=64 logits at task-relevant positions, then training a pruned student via KL divergence against those cached logits + hard CE on gold targets — recovers most of the F1 gap from structured tile pruning. Validated at Taylor 20% (F1: 0.097 → 0.471, recovered 72.3%) and Taylor 50% (F1: 0.000 → 0.337, recovered 54.8%) on Qwen 2.5 3B with CoNLL-2003 NER. r=16 LoRA, full-precision AdamW, 1500 training steps, 5 minutes per run, 7 GB peak VRAM, 3.3 MB cached teacher data on disk.

**Why it matters.** The constraint that "we can't fit two 3B models on 8 GB VRAM" was never the bottleneck for distillation. Top-K logit caching (1600× smaller than full-vocab logits) makes teacher-student training feasible on commodity hardware. Combined with structured tile pruning (Phase 1-4), this gives a complete pipeline: prune for speed → distill on task-relevant cached logits → recover task capability. End-to-end real on a single 8 GB laptop GPU.

**Generalization rule.** Distillation transfers what the teacher demonstrates in the training data — not general capability. Caching teacher logits over CoNLL annotations recovered NER. **MMLU stayed at random at both 20% and 50% sparsity post-distillation** — the teacher cache contained no facts about chemistry, medicine, etc., so those couldn't transfer. To recover specific factual knowledge requires a knowledge-rich teacher cache (Wikipedia, pretraining text); to recover general fluency requires CPT-style data.

**Counterintuitive sub-finding.** Heavier-pruned + distilled models generalize *better* across data than moderately-pruned ones. Taylor 20% post-distill had a 15.5pp dev-vs-test F1 gap; Taylor 50% had only 1.3pp. At moderate damage the LoRA finds dev-specific patterns to over-fit to; at heavy damage there's no residual base capability to exploit, so the LoRA learns general task structure that generalizes cleanly. The over-fitting risk is highest at moderate damage levels.

**What to take away.** Don't reject teacher distillation because of memory constraints — top-K caching makes it tractable on small VRAM. Match the teacher cache content to the capability you're trying to restore. A single LoRA training run can rescue a model from random-output state given the right signal.

---

## 8. Narrow layer duplication destroys MMLU; wide duplication is neutral

**Finding (Phase 5 negative result).** Across 13 duplication variants on Qwen 2.5 3B, every narrow duplication (1-11 layers in the circuit region) regressed MMLU by 3-14pp. The only variant that didn't regress — F stitched-top-9 at **26 layers (layers 9-34, 72% of the 36-layer stack)** — landed at 48.87% vs dense 48.67%, which is within the ±1.1pp noise floor.

**Why it matters.** The RYS thesis at 72B was "duplicate the 7-layer circuit in the middle and gain accuracy." At 3B we couldn't reproduce that with any narrow window. The only neutral result comes from duplicating most of the model — which by symmetry approaches "run the model twice" and should trivially be roughly safe.

**Mechanism.** Narrow mid-stream duplication feeds out-of-distribution hidden states to downstream layers. Layer X was trained to receive layer X-1's output; when a duplicated block runs, layer X on the second pass receives the duplicate-block's exit output instead, which is distributionally different. The corruption compounds through remaining layers.

**What to take away.** The "one importance map serves both pruning AND duplication" premise is only half-validated. Taylor importance tells you what's safe to prune (well-confirmed). It does NOT reliably identify narrow windows that are beneficial to duplicate, at least at this model size. If you want duplication benefits, you likely need a RYS-style systematic sweep over many (start, end) pairs rather than an importance-derived pick.

---

## 9. RYS's relative position scaling doesn't transfer across model sizes

**Finding.** RYS found optimal duplication at layers 45-51 of 80 on Qwen2-72B (positions 0.56-0.64). Scaling that to our 36-layer Qwen 2.5 3B by ratio gives layers 20-23 — which is the **trough** of our Taylor importance map, not the peak. The circuit peak on a 3B model is at layers 8-16, much earlier in the stack.

**Why it matters.** Smaller models push reasoning circuits earlier, closer to the encoders. "Duplicate the middle third" is model-size-dependent advice. Data-driven detection (Taylor importance) locates the peak for a given model; scaling rules from one model size don't.

**What to take away.** Relative position rules ("middle layers") are a weaker prior than people assume. Measure importance, don't scale positions.

---

## 10. Circuits are not just "where importance is high"

**Finding.** "Low-importance layers are safe to duplicate" seemed intuitive — if they contribute little, running them twice should matter little. Empirically, it's backwards. Duplicating the deepest Taylor-negative layers (H trough = layers 1-4, heavy encoders) tanked MMLU by 8.53pp — worse than duplicating the peak partially.

**Why.** Negative Taylor score doesn't mean "unused." It means the loss's sensitivity to those weights is low at the sample points we calibrated on. Those layers are doing heavy encoding work every downstream layer depends on. Running them twice corrupts the early representation space everything else was trained against.

**What to take away.** Taylor importance is a loss-sensitivity metric, not a capacity-utilization metric. Low-importance ≠ passthrough. Any intuition that leans on "this layer isn't doing much so duplicating it should be safe" needs to be verified empirically, not inferred from scores.

---

## 11. Narrow-data LoRA training overwrites priors rather than repairing circuits

**Finding.** Fine-tuning a Taylor-50%-pruned Qwen 2.5 3B with 500 MMLU-aux samples × 3 epochs, answer-token-only loss. Training loss dropped convincingly (6.27 → 1.51). Net MMLU went from 26.68% → 25.99% (−0.69pp) with wild per-subject swings: global_facts +12pp, professional_medicine −13.6pp.

**Why.** `cais/mmlu/auxiliary_train` is scraped from ARC, RACE, OpenBookQA, HellaSwag — not MMLU. 500 streamed samples are dominated by one source style. The model learned that source's answer-token distribution, helping subjects whose format matches and hurting those whose don't. The *direction* of learning is toward that narrow prior, not toward repairing the broken MLP circuits.

**What to take away.** Recovery data should match the distribution the damaged circuits were trained on (pretraining-like text with next-token CE), not the evaluation task. Answer-token-only supervision on a narrow MCQ corpus is a prior-replacement intervention, not a repair.

---

## 12. Methodological: isolate one variable at a time

**Finding.** At one point we changed LoRA rank (16 → 64) and optimizer (AdamW → AdamW8bit) together. The combined result was dramatic (−3.79pp) and I initially interpreted it as "bigger rank overfits the narrow training data — pivot to pretraining recovery." That conclusion was premature; the isolation test (rank 16 + AdamW8bit alone) showed the 8-bit optimizer was the entire cause, not rank.

**What to take away.** When two variables change at once and you get a surprising result, the first instinct should be isolation, not interpretation. One more experiment (−90 seconds) prevents weeks of pursuing the wrong hypothesis.

---

## 13. Methodological: read variant specs from code, not from chart interpretation

**Finding.** I spent significant analysis on a "razor-sharp circuit boundary at layers [8, 16]" story. It turned out the winning F variant was actually range [9, 34] — a 26-layer duplication. I had looked at the per-layer Taylor chart, visually identified the "peak 8-16," and assumed `stitched_top_n(9)` would return that range. It didn't. Layers 33 and 34 had higher Taylor scores than layers 8 and 16, so the top-9 span pulled the max index out to 34.

**What to take away.** When code computes a specification (indices, ranges, thresholds), always read the printed output of that computation before analyzing results. Don't substitute your mental model of what the code should produce. The one `print(VARIANTS)` line I should have checked was sitting in the notebook the whole time.

---

## 14. Load-once and configure-in-place is essential on tight VRAM

**Finding.** Holding two copies of a 3B model on an 8 GB card is impossible. The pattern that works: load the model once, cache any original weights you'll need to restore on CPU, and swap model state in-place between variants via utility functions (`configure_model`, `set_duplication`, `apply_masks`, `restore_weights`).

**Gotcha 1.** IPython retains references to "deleted" models via `sys.last_traceback` and the `Out[]` cache. `del m; torch.cuda.empty_cache()` isn't enough — you have to clear IPython state explicitly.

**Gotcha 2.** When using HuggingFace models with duplicated layers, you must also sync `config.layer_types` (per-layer attention-mask type list) with the new ModuleList length, otherwise `Qwen2Model.forward` will IndexError on duplicated positions.

**Gotcha 3.** PEFT + gradient checkpointing on a frozen base requires `model.enable_input_require_grads()` right after `get_peft_model()`. Without it, checkpoint-recompute silently retains the full activation stack and OOMs.

---

## 15. bf16 determinism is strong enough to trust, weak enough to surprise

**Finding.** Throughout the project, most evaluations were reproducible to the question across sessions. But at least one (notebook 7 running F=[8,16] getting 38.21% while notebook 6's F=[9,34] reproducibly gave 48.87%) created confusion that turned out to be a *specification* mismatch, not a numerics mismatch — different F ranges were being tested under the same name. Once the specs matched, numbers matched.

**What to take away.** Before blaming CUDA kernel autotune, SDPA backend selection, allocator state, or thermal throttling for inexplicable numerical divergence — verify the specifications are identical. Chasing kernel-level ghosts when the actual bug is "you were running a different experiment" is expensive.

---

## Summary table — top findings and how confident we are

| # | Finding | Confidence |
|---|---|---|
| 1 | Taylor > ActW > Frobenius for tile importance | High |
| 2 | Per-component-type z-normalization is essential | High |
| 3 | Accuracy has a knee vs sparsity; stronger models = lower knee | High |
| 4 | Kernel break-even is a sparsity threshold | High |
| 5 | Sparsity speedup scales with model size (1B→3B) | High |
| 6 | Tensor cores with waste beat CUDA cores in this regime | Medium-High |
| 7 | 8-bit AdamW unsafe for low-gradient recovery | High |
| 8 | Narrow dup hurts; wide dup is neutral on Qwen 3B | High, but negative |
| 9 | RYS position scaling doesn't transfer across sizes | Medium |
| 10 | Low Taylor score ≠ passthrough (encoders are active) | Medium-High |
| 11 | MCQ-style recovery overwrites priors, doesn't repair | Medium-High |
| 12 | Always isolate one variable at a time | Methodological |
| 13 | Read variant specs from output, not from charts | Methodological |
| 14 | Load-once pattern is essential on tight VRAM | High |
| 15 | Verify specifications before blaming numerics | Methodological |

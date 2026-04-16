"""
Step 0: Load Gemma 3 1B unmodified and run the baseline MMLU evaluation.

Usage:
    python run_baseline.py
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from eval_mmlu import evaluate, print_results, save_results


def main():
    print(f"Model:  {config.MODEL_NAME}")
    print(f"Device: {config.DEVICE}")
    print(f"Dtype:  {config.TORCH_DTYPE}")
    print()

    # ── Load model ─────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    print("Loading model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=getattr(torch, config.TORCH_DTYPE),
        device_map=config.DEVICE,
    )
    model.eval()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # ── Quick sanity check ─────────────────────────────────────────────
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e9:.2f}B")

    mem_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"GPU memory: {mem_gb:.2f} GB")
    print()

    # ── Run MMLU eval ──────────────────────────────────────────────────
    print("Running baseline MMLU evaluation...")
    t0 = time.time()
    results = evaluate(model, tokenizer, tag="baseline")
    eval_time = time.time() - t0

    results["meta"] = {
        "load_time_s": round(load_time, 1),
        "eval_time_s": round(eval_time, 1),
        "gpu_memory_gb": round(mem_gb, 2),
        "n_params": n_params,
    }

    print_results(results)
    save_results(results)
    print(f"Evaluation completed in {eval_time:.1f}s")


if __name__ == "__main__":
    main()

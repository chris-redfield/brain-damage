"""
MMLU subset evaluation via log-probability scoring.

For each question we compute the log-probability of each answer choice
(A/B/C/D) and pick the highest. This is deterministic and doesn't require
generation.
"""

import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


# ── MMLU formatting ────────────────────────────────────────────────────────

CHOICE_LABELS = ["A", "B", "C", "D"]


def format_mmlu_prompt(question: str, choices: list[str]) -> str:
    """Format a single MMLU question as a multiple-choice prompt."""
    lines = [question]
    for label, choice in zip(CHOICE_LABELS, choices):
        lines.append(f"{label}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


# ── Log-prob scoring ───────────────────────────────────────────────────────


def score_choices(
    model,
    tokenizer,
    prompts: list[str],
    device: str = config.DEVICE,
) -> list[int]:
    """
    For each prompt, compute log-prob of each answer token (A/B/C/D) and
    return the index of the highest-scoring choice.
    """
    # Pre-tokenize the answer tokens — we only need a single token per choice
    choice_token_ids = []
    for label in CHOICE_LABELS:
        ids = tokenizer.encode(label, add_special_tokens=False)
        choice_token_ids.append(ids[-1])  # take the last token (the letter)

    predictions = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        # logits at the last position → distribution over next token
        last_logits = outputs.logits[0, -1, :]  # (vocab_size,)
        log_probs = torch.log_softmax(last_logits, dim=-1)

        choice_scores = [log_probs[tid].item() for tid in choice_token_ids]
        predictions.append(int(torch.tensor(choice_scores).argmax()))

    return predictions


# ── Dataset loading ────────────────────────────────────────────────────────


def load_mmlu_subset(
    subjects: list[str] | None = None,
    split: str = config.MMLU_SPLIT,
) -> dict[str, list[dict]]:
    """
    Load MMLU subjects from HuggingFace datasets.

    Returns {subject_name: [{"question", "choices", "answer"}, ...]}.
    """
    subjects = subjects or config.MMLU_SUBJECTS

    data_by_subject: dict[str, list[dict]] = {}

    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split=split, trust_remote_code=True)
        examples = []
        for row in ds:
            examples.append(
                {
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": int(row["answer"]),
                }
            )
        data_by_subject[subject] = examples

    return data_by_subject


# ── Evaluation loop ───────────────────────────────────────────────────────


def evaluate(
    model,
    tokenizer,
    subjects: list[str] | None = None,
    tag: str = "baseline",
) -> dict:
    """
    Run MMLU subset evaluation and return results dict.
    """
    data = load_mmlu_subset(subjects)

    results_per_subject = {}
    total_correct = 0
    total_count = 0

    for subject, examples in tqdm(data.items(), desc=f"[{tag}] MMLU subjects"):
        prompts = [
            format_mmlu_prompt(ex["question"], ex["choices"]) for ex in examples
        ]
        answers = [ex["answer"] for ex in examples]

        preds = score_choices(model, tokenizer, prompts)

        correct = sum(p == a for p, a in zip(preds, answers))
        accuracy = correct / len(answers) if answers else 0.0

        results_per_subject[subject] = {
            "correct": correct,
            "total": len(answers),
            "accuracy": round(accuracy, 4),
        }
        total_correct += correct
        total_count += len(answers)

    overall_accuracy = total_correct / total_count if total_count else 0.0

    return {
        "tag": tag,
        "model": config.MODEL_NAME,
        "subjects": results_per_subject,
        "overall": {
            "correct": total_correct,
            "total": total_count,
            "accuracy": round(overall_accuracy, 4),
        },
    }


def save_results(results: dict, filename: str | None = None):
    """Save results dict as JSON."""
    out_dir = Path(config.RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"mmlu_{results['tag']}.json"

    path = out_dir / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")
    return path


# ── Standalone entry point ────────────────────────────────────────────────


def print_results(results: dict):
    """Pretty-print results to stdout."""
    tag = results["tag"]
    print(f"\n{'=' * 60}")
    print(f"  MMLU Results — {tag}")
    print(f"{'=' * 60}")

    for subject, stats in sorted(results["subjects"].items()):
        acc = stats["accuracy"] * 100
        bar = "█" * int(acc // 5) + "░" * (20 - int(acc // 5))
        print(f"  {subject:<30s} {bar} {acc:5.1f}% ({stats['correct']}/{stats['total']})")

    overall = results["overall"]
    print(f"{'─' * 60}")
    print(f"  {'OVERALL':<30s}               {overall['accuracy'] * 100:5.1f}% ({overall['correct']}/{overall['total']})")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    print(f"Loading model: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=getattr(torch, config.TORCH_DTYPE),
        device_map=config.DEVICE,
    )
    model.eval()

    results = evaluate(model, tokenizer, tag="baseline")
    print_results(results)
    save_results(results)

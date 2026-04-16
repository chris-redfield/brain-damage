"""
Project configuration for Regional Magnitude Pruning experiments.
"""

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = "google/gemma-3-1b-it"  # instruction-tuned, ~2 GB in bf16
TORCH_DTYPE = "bfloat16"
DEVICE = "cuda"

# ---------------------------------------------------------------------------
# MMLU evaluation
# ---------------------------------------------------------------------------
# Fixed subset of MMLU subjects — diverse enough to be meaningful, small
# enough to run fast (~500 questions total).
MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "college_chemistry",
    "college_computer_science",
    "econometrics",
    "global_facts",
    "machine_learning",
    "moral_scenarios",
    "professional_medicine",
    "us_foreign_policy",
]
MMLU_SPLIT = "test"
MMLU_BATCH_SIZE = 8  # per-device batch size for log-prob scoring

# ---------------------------------------------------------------------------
# Tile / block pruning
# ---------------------------------------------------------------------------
TILE_SIZES = [(128, 128)]  # (rows, cols) — validated as best granularity
PRUNING_TARGETS = [0.10, 0.20, 0.30, 0.50]  # fraction of tiles to zero out

# Only prune MLP projections for now. Attention projections (Q/K/V/O) are
# tightly coupled and risky to tile-prune — experiment with them later.
PRUNE_TARGETS_PATTERNS = ["gate_proj", "up_proj", "down_proj"]
PRUNE_SKIP_PATTERNS = ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj",
                        "self_attn", "embed_tokens", "lm_head"]
MAX_PRUNE_PER_MATRIX = 0.50  # never prune more than 50% of any single matrix
PRUNE_SKIP_LAYERS = []  # actw scoring naturally protects critical layers

# ---------------------------------------------------------------------------
# Layer duplication (RYS-style)
# ---------------------------------------------------------------------------
# Will be populated after importance scoring; these are the layer index
# ranges to try duplicating (inclusive start, exclusive end).
DUPLICATION_RANGES = []  # e.g. [(8, 14), (10, 16)]

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
CALIBRATION_DATASET = "allenai/c4"
CALIBRATION_SUBSET = "en"
CALIBRATION_SAMPLES = 1024
CALIBRATION_SEQ_LEN = 512

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"

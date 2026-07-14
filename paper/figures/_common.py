"""
paper/figures/_common.py — shared loaders and the interim analytic SE.

All figure scripts read from the existing pipeline outputs only — no API
calls, no IRT refit. The analytic SE here is the standard delta-method
standard error for a single-item logit difficulty,

    SE(b) = 1 / sqrt(N * P * (1 - P)),   P = 1 / (1 + e^b),   N = 45,

which is the interim uncertainty reported in the paper until the raw
response matrix (logs/arena_runs/phase1_baseline/responses.jsonl) is
regenerated and py-irt posterior SDs (evaluator/pyirt_fit.py) replace it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

# Repo root = two levels up from this file (paper/figures/_common.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "evaluator" / "output"
FROZEN_BANK = OUTPUT_DIR / "phase2_frozen_bank.jsonl"
DELTA_THETA = OUTPUT_DIR / "phase3a_strictness_deltatheta.json"

# Phase 1 grid: 3 models x 5 temps x 3 strictness = 45 virtual examinees.
N_EXAMINEES = 45

# Consistent, colourblind-safe palette used across all figures.
PALETTE = {
    "llama-3.3-70b-versatile": "#0072B2",
    "openai/gpt-oss-20b": "#E69F00",
    "llama-3.1-8b-instant": "#009E73",
}
MODEL_SHORT = {
    "llama-3.3-70b-versatile": "llama-3.3-70b",
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "llama-3.1-8b-instant": "llama-3.1-8b",
}


def load_frozen_bank(path: Path = FROZEN_BANK) -> list[dict]:
    """Read phase2_frozen_bank.jsonl -> list of {task_id, b, point_biserial, infit, outfit}."""
    if not path.exists():
        raise FileNotFoundError(
            f"frozen bank not found at {path}; run run_phase2.py to (re)create it"
        )
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_delta_theta(path: Path = DELTA_THETA) -> dict:
    """Read phase3a_strictness_deltatheta.json."""
    if not path.exists():
        raise FileNotFoundError(f"delta-theta json not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def analytic_se_b(b: float, n: int = N_EXAMINEES) -> float:
    """
    Delta-method SE of a one-item logit difficulty.

    P = 1/(1+e^b); SE(b) = 1/sqrt(N * P * (1-P)). Returned in logits.
    """
    p = 1.0 / (1.0 + math.exp(b))
    info = n * p * (1.0 - p)
    if info <= 0.0:
        return float("inf")
    return 1.0 / math.sqrt(info)


def apply_paper_style() -> None:
    """Uniform matplotlib styling for all figures (no LaTeX dependency)."""
    import matplotlib

    matplotlib.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.autolayout": True,
        }
    )

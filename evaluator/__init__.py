"""
Evaluator layer — OFFLINE psychometrics engine.

Reads jsonl response logs produced by /arena/, builds the binary response
matrix, and fits IRT models (1PL Rasch and 2PL) using girth.

CRITICAL ENGINEERING RULE 2 — this layer MUST NEVER call an LLM API.
All inputs are jsonl files written by the Arena. The isolation is
enforced mechanically by tests/test_evaluator_isolation.py which
asserts that no module under evaluator/ transitively imports openai,
groq, harness, or arena.runner.
"""

from evaluator.response_matrix import (
    ResponseMatrix,
    build_response_matrix,
    load_jsonl,
)
from evaluator.rasch import RaschFit, fit_rasch, anchor_baseline_to_zero
from evaluator.twopl import TwoPLFit, fit_twopl
from evaluator.qc import QCReport, run_2pl_qc

__all__ = [
    "ResponseMatrix",
    "build_response_matrix",
    "load_jsonl",
    "RaschFit",
    "fit_rasch",
    "anchor_baseline_to_zero",
    "TwoPLFit",
    "fit_twopl",
    "QCReport",
    "run_2pl_qc",
]

"""
evaluator/twopl.py — 2PL calibration via girth.

Phase 2 deliverable: estimate both item difficulty `b` and item
discrimination `a`. Items with a ≤ 0 indicate broken ground truth
(more-able examinees fail more often than less-able ones) and are
flagged for removal by `evaluator.qc.run_2pl_qc`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from girth import twopl_mml

from evaluator.response_matrix import ResponseMatrix


@dataclass
class TwoPLFit:
    """2PL calibration result aligned to a ResponseMatrix."""

    item_ids: list[str]
    a: np.ndarray  # discrimination, shape (n_items,)
    b: np.ndarray  # difficulty,     shape (n_items,)


def _prepare_dataset(rm: ResponseMatrix) -> np.ndarray:
    mat = rm.matrix.copy()
    mat[mat < 0] = 0
    return mat.astype(np.int8)


def fit_twopl(rm: ResponseMatrix) -> TwoPLFit:
    dataset = _prepare_dataset(rm)
    out = twopl_mml(dataset)

    # girth versions vary: tuple or dict
    if isinstance(out, dict):
        a = np.asarray(out.get("Discrimination")).flatten()
        b = np.asarray(out.get("Difficulty")).flatten()
    else:
        a, b = out
        a = np.asarray(a).flatten()
        b = np.asarray(b).flatten()

    return TwoPLFit(item_ids=list(rm.item_ids), a=a, b=b)

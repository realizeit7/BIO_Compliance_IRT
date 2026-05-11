"""
evaluator/twopl.py — 2PL calibration via girth.

Phase 2 deliverable: estimate both item difficulty `b` and item
discrimination `a`. Items with low or non-finite `a` are flagged for
removal by `evaluator.qc.run_2pl_qc`.

Uses `twopl_jml` (joint MLE) rather than `twopl_mml` (marginal MLE):
on our 45-examinee grid, MML pins every item at the discrimination
upper bound (a=5.0) — JML produces a real spread (a roughly in
[0.25, 4.0]) because it co-estimates the 45 examinee thetas instead of
integrating over a population.

All-pass / all-fail items have undefined 2PL discrimination; they are
returned with a=b=NaN so the QC layer drops them via its non-finite
branch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from girth import twopl_jml

from evaluator.response_matrix import ResponseMatrix


@dataclass
class TwoPLFit:
    """2PL calibration result aligned to a ResponseMatrix."""

    item_ids: list[str]
    a: np.ndarray  # discrimination, shape (n_items,)
    b: np.ndarray  # difficulty,     shape (n_items,)


def fit_twopl(rm: ResponseMatrix) -> TwoPLFit:
    mat = rm.matrix.copy()
    mat[mat < 0] = 0
    mat = mat.astype(np.int32)

    n_examinees = mat.shape[1]
    row_sums = mat.sum(axis=1)
    variable = (row_sums > 0) & (row_sums < n_examinees)

    a_full = np.full(mat.shape[0], np.nan)
    b_full = np.full(mat.shape[0], np.nan)

    if variable.any():
        out = twopl_jml(mat[variable])
        if isinstance(out, dict):
            a = np.asarray(out.get("Discrimination")).flatten()
            b = np.asarray(out.get("Difficulty")).flatten()
        else:
            a, b = out
            a = np.asarray(a).flatten()
            b = np.asarray(b).flatten()
        a_full[variable] = a
        b_full[variable] = b

    return TwoPLFit(item_ids=list(rm.item_ids), a=a_full, b=b_full)

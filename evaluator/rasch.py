"""
evaluator/rasch.py — 1PL (Rasch) calibration via girth.

Phase 1 deliverable: estimate item difficulty `b` for every item from the
binary response matrix. The returned `b` array assumes the marginal
ability distribution is N(0, 1), which is girth's default assumption.

To satisfy the "Fix the baseline at θ=0" mandate, callers should pass the
fitted b's plus the baseline examinee's response vector to
`anchor_baseline_to_zero`, which post-hoc shifts b so that the baseline
examinee's MLE θ equals zero exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from girth import ability_mle, rasch_mml

from evaluator.response_matrix import ResponseMatrix


@dataclass
class RaschFit:
    """1PL calibration result aligned to a ResponseMatrix."""

    item_ids: list[str]
    b: np.ndarray  # shape (n_items,), item difficulties
    baseline_examinee_id: str | None = None
    theta_baseline_raw: float | None = None  # MLE θ before anchoring (if anchored)
    anchored: bool = False


def _prepare_dataset(rm: ResponseMatrix) -> np.ndarray:
    """girth requires bool / int with no missing values. Replace -1 with 0."""
    mat = rm.matrix.copy()
    # -1 sentinel from build_response_matrix means "no observation" — treat as
    # missing-at-random failure. In Phase 1 the runner produces a complete grid
    # so this branch should not fire; we map to 0 conservatively.
    mat[mat < 0] = 0
    return mat.astype(np.int8)


def fit_rasch(rm: ResponseMatrix) -> RaschFit:
    """Fit a 1PL Rasch model with girth's MML estimator."""
    dataset = _prepare_dataset(rm)
    b = rasch_mml(dataset)
    if isinstance(b, dict):  # girth has returned dict in some versions
        b = b.get("Difficulty", next(iter(b.values())))
    b = np.asarray(b).flatten()
    return RaschFit(item_ids=list(rm.item_ids), b=b)


def anchor_baseline_to_zero(
    fit: RaschFit,
    rm: ResponseMatrix,
    baseline_examinee_id: str,
) -> RaschFit:
    """
    Shift `b` so that the baseline examinee's MLE θ equals 0.

    Mechanism: in the 1PL model, P(correct | θ, b) = σ(θ − b). Shifting all
    b by a constant c is equivalent to shifting all θ by the same c. Pick
    c = θ_baseline so that θ_baseline_anchored = 0 and b_anchored = b − c.
    """
    baseline_responses = rm.baseline_responses(baseline_examinee_id)
    # ability_mle requires items × participants; we pass a single column
    single = baseline_responses.reshape(-1, 1)
    discrimination = np.ones_like(fit.b)
    theta = ability_mle(single, fit.b, discrimination)
    theta_scalar = float(np.atleast_1d(theta)[0])

    # If the baseline aced or zeroed every item, MLE is undefined (±inf or NaN).
    # In that case skip anchoring and surface the situation to the caller.
    if not np.isfinite(theta_scalar):
        return RaschFit(
            item_ids=fit.item_ids,
            b=fit.b.copy(),
            baseline_examinee_id=baseline_examinee_id,
            theta_baseline_raw=theta_scalar,
            anchored=False,
        )

    return RaschFit(
        item_ids=fit.item_ids,
        b=fit.b - theta_scalar,
        baseline_examinee_id=baseline_examinee_id,
        theta_baseline_raw=theta_scalar,
        anchored=True,
    )

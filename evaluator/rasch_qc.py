"""
evaluator/rasch_qc.py — Phase 2 item QC under the Rasch (1PL) model.

The Rasch model itself doesn't expose a per-item discrimination, so QC
relies on three classical signals computed on the response matrix:

- **Zero-variance**: items with all-pass or all-fail responses; Rasch b
  is at the bound and the item carries no information. Drop unconditionally.
- **Point-biserial correlation**: correlation between the item response
  vector and the total examinee score (corrected: total minus this item).
  Negative pb indicates an inverted item — smarter examinees fail more
  than weaker ones — which is the same broken-ground-truth signal the
  2PL `a ≤ 0` filter was meant to catch.
- **Infit / Outfit**: Rasch fit statistics. Both are mean squared
  standardized residuals (Z² where Z = (X − P)/sqrt(P(1−P))). Outfit is
  unweighted; Infit is information-weighted (weighted by P(1−P), so
  observations near the item's b matter more). Linacre's standard
  "productive measurement" range is [0.5, 1.5], but that is calibrated
  for thousands of examinees. On our 45-cell grid we use asymmetric
  defaults — upper bound only — because "too deterministic" items
  (low MSR) separate examinees cleanly, which is desirable for a Δθ
  bank, not a defect.

Examinee θ for the residual computation is estimated via girth's MLE
(matches Linacre/WINSTEPS practice for Rasch fit stats). Examinees with
non-finite MLE θ (would happen only at all-pass / all-fail score
patterns, which doesn't occur on our 45-cell grid) are dropped from
the residual computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from girth import ability_mle

from evaluator.rasch import RaschFit
from evaluator.response_matrix import ResponseMatrix


@dataclass
class RaschQCReport:
    """Outcome of Rasch-based Phase 2 QC."""

    healthy_item_ids: list[str]
    dropped_item_ids: list[str]
    dropped_reasons: dict[str, str]
    healthy_b: np.ndarray
    healthy_pb: np.ndarray
    healthy_infit: np.ndarray
    healthy_outfit: np.ndarray
    # Full per-item diagnostics (aligned to fit.item_ids), so callers can
    # inspect what dropped items looked like before they were removed.
    all_pb: np.ndarray = field(default_factory=lambda: np.array([]))
    all_infit: np.ndarray = field(default_factory=lambda: np.array([]))
    all_outfit: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> dict:
        n_total = len(self.healthy_item_ids) + len(self.dropped_item_ids)
        return {
            "n_healthy": len(self.healthy_item_ids),
            "n_dropped": len(self.dropped_item_ids),
            "drop_rate": len(self.dropped_item_ids) / max(1, n_total),
            "median_b_healthy": _safe_median(self.healthy_b),
            "median_pb_healthy": _safe_median(self.healthy_pb),
            "median_infit_healthy": _safe_median(self.healthy_infit),
            "median_outfit_healthy": _safe_median(self.healthy_outfit),
        }


def _safe_median(arr: np.ndarray) -> float:
    return float(np.median(arr)) if len(arr) else float("nan")


def _corrected_point_biserial(mat: np.ndarray) -> np.ndarray:
    """Per-item point-biserial vs (total score − this item)."""
    n_items = mat.shape[0]
    total = mat.sum(axis=0)  # (n_examinees,)
    pb = np.full(n_items, np.nan)
    for i in range(n_items):
        rest = total - mat[i]
        if mat[i].std() == 0 or rest.std() == 0:
            continue
        pb[i] = float(np.corrcoef(mat[i], rest)[0, 1])
    return pb


def _infit_outfit(mat: np.ndarray, b: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rasch infit/outfit. mat: (n_items, n_examinees), b: (n_items,), theta: (n_examinees,)."""
    # Probabilities under Rasch: P_ij = sigmoid(theta_j - b_i)
    logits = theta[None, :] - b[:, None]  # (n_items, n_examinees)
    P = 1.0 / (1.0 + np.exp(-logits))
    P = np.clip(P, 1e-9, 1 - 1e-9)
    var = P * (1.0 - P)
    resid = mat - P
    z2 = resid * resid / var

    outfit = z2.mean(axis=1)  # unweighted MSR
    # Infit: information-weighted MSR. Weight by var so well-targeted
    # observations dominate. Algebraically: sum(resid^2)/sum(var).
    infit_num = (resid * resid).sum(axis=1)
    infit_den = var.sum(axis=1)
    infit = np.where(infit_den > 0, infit_num / infit_den, np.nan)
    return infit, outfit


def run_rasch_qc(
    fit: RaschFit,
    rm: ResponseMatrix,
    *,
    pb_threshold: float = 0.0,
    infit_range: tuple[float, float] = (0.0, 1.5),
    outfit_range: tuple[float, float] = (0.0, 2.0),
) -> RaschQCReport:
    """
    Drop items that fail any of:
      - zero variance (all-pass or all-fail)
      - point-biserial ≤ pb_threshold (default 0.0 → drop only inverted/flat)
      - infit outside infit_range (default (0, 1.5] — upper bound only)
      - outfit outside outfit_range (default (0, 2.0] — upper bound only;
        outfit is more outlier-sensitive on small examinee samples)
    """
    mat = rm.matrix.copy()
    mat[mat < 0] = 0
    mat = mat.astype(np.float64)
    n_examinees = mat.shape[1]

    discrimination = np.ones_like(fit.b)
    theta = ability_mle(mat.astype(np.int32), fit.b, discrimination)
    theta = np.asarray(theta).flatten()

    # Drop non-finite examinees from the residual computation so a single
    # all-pass / all-fail examinee can't NaN out every item's infit.
    finite_ex = np.isfinite(theta)
    pb = _corrected_point_biserial(mat)
    infit, outfit = _infit_outfit(mat[:, finite_ex], fit.b, theta[finite_ex])

    healthy_ids: list[str] = []
    dropped_ids: list[str] = []
    reasons: dict[str, str] = {}
    healthy_b: list[float] = []
    healthy_pb: list[float] = []
    healthy_infit: list[float] = []
    healthy_outfit: list[float] = []

    row_sums = mat.sum(axis=1)

    for i, tid in enumerate(fit.item_ids):
        if row_sums[i] == 0 or row_sums[i] == n_examinees:
            dropped_ids.append(tid)
            reasons[tid] = "zero_variance (all-pass or all-fail item)"
            continue
        if not np.isfinite(fit.b[i]):
            dropped_ids.append(tid)
            reasons[tid] = f"non_finite_b (b={fit.b[i]})"
            continue
        if not np.isfinite(pb[i]) or pb[i] <= pb_threshold:
            dropped_ids.append(tid)
            reasons[tid] = f"low_point_biserial (pb={pb[i]:.3f} <= {pb_threshold})"
            continue
        if not (infit_range[0] <= infit[i] <= infit_range[1]):
            dropped_ids.append(tid)
            reasons[tid] = f"infit_out_of_range (infit={infit[i]:.3f} not in {infit_range})"
            continue
        if not (outfit_range[0] <= outfit[i] <= outfit_range[1]):
            dropped_ids.append(tid)
            reasons[tid] = f"outfit_out_of_range (outfit={outfit[i]:.3f} not in {outfit_range})"
            continue
        healthy_ids.append(tid)
        healthy_b.append(float(fit.b[i]))
        healthy_pb.append(float(pb[i]))
        healthy_infit.append(float(infit[i]))
        healthy_outfit.append(float(outfit[i]))

    return RaschQCReport(
        healthy_item_ids=healthy_ids,
        dropped_item_ids=dropped_ids,
        dropped_reasons=reasons,
        healthy_b=np.array(healthy_b),
        healthy_pb=np.array(healthy_pb),
        healthy_infit=np.array(healthy_infit),
        healthy_outfit=np.array(healthy_outfit),
        all_pb=pb,
        all_infit=infit,
        all_outfit=outfit,
    )

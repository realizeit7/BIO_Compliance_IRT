"""
evaluator/qc.py — Phase 2 item quality control.

Drops items with non-positive 2PL discrimination (a ≤ 0). A non-positive
discrimination means the item function is flat or inverted: more-able
examinees are not more likely to pass, which indicates broken ground
truth, an ambiguous question, or judge error correlated with examinee
type. These items contaminate IRT estimation downstream and must be
quarantined before the holdout bank is frozen for Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluator.twopl import TwoPLFit


@dataclass
class QCReport:
    """Outcome of Phase 2 discrimination QC."""

    healthy_item_ids: list[str]
    dropped_item_ids: list[str]
    dropped_reasons: dict[str, str]
    healthy_a: np.ndarray
    healthy_b: np.ndarray

    def summary(self) -> dict:
        return {
            "n_healthy": len(self.healthy_item_ids),
            "n_dropped": len(self.dropped_item_ids),
            "drop_rate": (
                len(self.dropped_item_ids)
                / max(1, len(self.healthy_item_ids) + len(self.dropped_item_ids))
            ),
            "median_a_healthy": float(np.median(self.healthy_a))
            if len(self.healthy_a)
            else float("nan"),
            "median_b_healthy": float(np.median(self.healthy_b))
            if len(self.healthy_b)
            else float("nan"),
        }


def run_2pl_qc(fit: TwoPLFit, *, a_threshold: float = 0.0) -> QCReport:
    """
    Flag items where a <= a_threshold (default 0.0 = strict negative-or-zero
    discrimination drop). Tighter thresholds (e.g., 0.3) are common in
    psychometric practice but require more data to be reliable.
    """
    healthy_ids: list[str] = []
    dropped_ids: list[str] = []
    reasons: dict[str, str] = {}
    healthy_a: list[float] = []
    healthy_b: list[float] = []

    for tid, a, b in zip(fit.item_ids, fit.a, fit.b):
        if not np.isfinite(a) or not np.isfinite(b):
            dropped_ids.append(tid)
            reasons[tid] = "zero_variance (all-pass or all-fail item, undefined 2PL)"
            continue
        if a <= a_threshold:
            dropped_ids.append(tid)
            reasons[tid] = f"low_discrimination (a={a:.3f} <= {a_threshold})"
            continue
        healthy_ids.append(tid)
        healthy_a.append(float(a))
        healthy_b.append(float(b))

    return QCReport(
        healthy_item_ids=healthy_ids,
        dropped_item_ids=dropped_ids,
        dropped_reasons=reasons,
        healthy_a=np.array(healthy_a),
        healthy_b=np.array(healthy_b),
    )

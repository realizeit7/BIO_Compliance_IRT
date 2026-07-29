"""
gen_fig2_item_fit.py — Figure 2: item-fit scatter (infit vs. outfit).

Reads evaluator/output/phase2_frozen_bank.jsonl and plots infit against
outfit for the healthy items, with the QC upper bounds (infit 1.5, outfit
2.0) drawn in. All healthy items lie inside the bounds by construction;
the figure shows how far inside (i.e. how clean the fit is).
Output: paper/figures/fig2_item_fit.pdf (and .png).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _common import apply_paper_style, load_frozen_bank


def main() -> None:
    apply_paper_style()
    import matplotlib.pyplot as plt

    rows = load_frozen_bank()
    infit = np.array([r["infit"] for r in rows], dtype=float)
    outfit = np.array([r["outfit"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.scatter(infit, outfit, s=10, c="#009E73", alpha=0.45, edgecolors="none")

    # QC upper bounds applied in run_phase2.py.
    ax.axvline(1.5, color="#D55E00", linestyle="--", linewidth=1.2, label="infit QC = 1.5")
    ax.axhline(2.0, color="#CC79A7", linestyle="--", linewidth=1.2, label="outfit QC = 2.0")
    # Ideal Rasch fit = 1.0 on both axes.
    ax.axvline(1.0, color="grey", linestyle=":", linewidth=0.8)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=0.8)

    ax.set_xlabel("infit (information-weighted MSR)")
    ax.set_ylabel("outfit (unweighted MSR)")
    ax.set_title(f"Item fit, frozen bank ($n={len(rows)}$)")
    # Opaque frame: the scatter reaches into the upper-left corner, and
    # without it a point sits behind the legend text.
    ax.legend(loc="upper left", frameon=True, framealpha=0.92,
              facecolor="white", edgecolor="0.8").set_zorder(10)

    out = Path(__file__).resolve().parent / "fig2_item_fit"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    print(f"[fig2] wrote {out}.pdf / .png  (median infit={np.median(infit):.3f}, "
          f"median outfit={np.median(outfit):.3f})")


if __name__ == "__main__":
    main()

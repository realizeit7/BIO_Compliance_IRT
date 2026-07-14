"""
gen_fig1_b_hist.py — Figure 1: difficulty (b) distribution of the frozen bank.

Reads evaluator/output/phase2_frozen_bank.jsonl and plots a histogram of the
anchored Rasch difficulties for the 1,284 healthy items, with the median
marked. Output: paper/figures/fig1_b_hist.pdf (and .png).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _common import apply_paper_style, load_frozen_bank


def main() -> None:
    apply_paper_style()
    import matplotlib.pyplot as plt

    rows = load_frozen_bank()
    b = np.array([r["b"] for r in rows], dtype=float)
    median_b = float(np.median(b))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.hist(b, bins=40, color="#0072B2", alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(median_b, color="#D55E00", linestyle="--", linewidth=1.5,
               label=f"median $b = {median_b:+.2f}$")
    ax.axvline(0.0, color="grey", linestyle=":", linewidth=1.0, label=r"$\theta=0$ anchor")

    ax.set_xlabel(r"item difficulty $b$ (logits, baseline-anchored)")
    ax.set_ylabel("number of items")
    ax.set_title(f"Frozen-bank difficulty distribution ($n={len(b)}$)")
    ax.legend(loc="upper left")

    out = Path(__file__).resolve().parent / "fig1_b_hist"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    print(f"[fig1] wrote {out}.pdf / .png  (n={len(b)}, median b={median_b:+.3f}, "
          f"min={b.min():+.3f}, max={b.max():+.3f})")


if __name__ == "__main__":
    main()

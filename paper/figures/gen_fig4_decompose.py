"""
gen_fig4_decompose.py — Figure 4: format-unlock vs. reasoning decomposition.

For each subject model, the strict-judge Δθ (none→strict) decomposes into:
  * reasoning gain  = lenient-judge Δθ (conclusion + reasoning only)
  * format unlock   = strict Δθ − lenient Δθ (the citation-format component)

Reads evaluator/output/phase3a_strictness_deltatheta.json and draws stacked
horizontal bars. Output: paper/figures/fig4_decompose.pdf (and .png).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _common import MODEL_SHORT, apply_paper_style, load_delta_theta


def main() -> None:
    apply_paper_style()
    import matplotlib.pyplot as plt

    data = load_delta_theta()
    strict = data["delta_theta_vs_none_strict"]
    lenient = data["delta_theta_vs_none_lenient"]
    fmt = data["format_unlock_delta"]
    models = data["models"]

    reasoning = np.array([lenient[m]["strict"] for m in models])
    format_unlock = np.array([fmt[m]["strict"] for m in models])
    labels = [MODEL_SHORT.get(m, m) for m in models]
    y = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    # The reasoning segment is drawn last (higher zorder). When it is
    # negative the format segment starts left of zero and would otherwise
    # paint straight over it, hiding the reasoning component entirely.
    ax.barh(y, format_unlock, left=reasoning, color="#E69F00", zorder=2,
            label="format unlock (strict $-$ lenient)")
    ax.barh(y, reasoning, color="#0072B2", zorder=3,
            label="reasoning gain (lenient $\\Delta\\theta$)")

    handles, lab = ax.get_legend_handles_labels()
    order = [1, 0]  # reasoning first, matching the left-to-right reading order

    ax.axvline(0.0, color="black", linewidth=0.8, zorder=4)

    # Fix the x-limits before annotating so that "does this label fit inside
    # its segment?" is decided against the final axis span, not a provisional
    # one. Pad the right so end-of-bar labels stay inside the axes.
    lo = min(0.0, float((reasoning + np.minimum(format_unlock, 0)).min()))
    hi = float((reasoning + np.maximum(format_unlock, 0)).max())
    span = hi - lo
    ax.set_xlim(lo - 0.16 * span, hi + 0.08 * span)
    span = ax.get_xlim()[1] - ax.get_xlim()[0]

    # A label only goes inside its own segment when the segment is wide
    # enough to hold it. Otherwise it is placed just past the outer edge of
    # the whole bar, which keeps it clear of the axis line and of the
    # neighbouring segment.
    min_inside = 0.11 * span
    pad = 0.012 * span

    for yi, (r, f) in enumerate(zip(reasoning, format_unlock)):
        # Reasoning segment: anchored at zero, so a label that does not fit
        # goes immediately to its left, where nothing else is drawn.
        if abs(r) >= min_inside:
            ax.text(r / 2, yi, f"{r:+.1f}", va="center", ha="center",
                    fontsize=8, color="white", zorder=5)
        elif abs(r) > 0.02:
            ax.text(min(0.0, r) - pad, yi, f"{r:+.1f}", va="center",
                    ha="right", fontsize=8, color="#0072B2", zorder=5)

        # Format segment: a label that does not fit goes past the outer end.
        if abs(f) >= min_inside:
            ax.text(r + f / 2, yi, f"{f:+.1f}", va="center", ha="center",
                    fontsize=8, color="black", zorder=5)
        elif abs(f) > 0.02:
            ax.text(max(0.0, r + f) + pad, yi, f"{f:+.1f}", va="center",
                    ha="left", fontsize=8, color="#8a6100", zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\Delta\theta$ at strict prompt (logits)")
    ax.set_title("What system-prompt strictness buys: format vs. reasoning",
                 pad=26)
    # The legend sits above the axes. Inside the axes it overlapped the
    # longest bar and collided with that bar's value label.
    ax.legend([handles[i] for i in order], [lab[i] for i in order],
              loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
              frameon=False, borderaxespad=0.0, handlelength=1.4,
              columnspacing=1.4, fontsize=8)

    out = Path(__file__).resolve().parent / "fig4_decompose"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    print(f"[fig4] wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()

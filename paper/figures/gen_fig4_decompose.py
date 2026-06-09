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

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.barh(y, reasoning, color="#0072B2", label="reasoning gain (lenient $\\Delta\\theta$)")
    ax.barh(y, format_unlock, left=reasoning, color="#E69F00",
            label="format unlock (strict $-$ lenient)")

    # Annotate each segment with its value.
    for yi, (r, f) in enumerate(zip(reasoning, format_unlock)):
        if abs(r) > 0.15:
            ax.text(r / 2, yi, f"{r:+.1f}", va="center", ha="center", fontsize=8, color="white")
        if abs(f) > 0.15:
            ax.text(r + f / 2, yi, f"{f:+.1f}", va="center", ha="center", fontsize=8, color="black")

    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\Delta\theta$ at strict prompt (logits)")
    ax.set_title("What system-prompt strictness buys: format vs. reasoning")
    ax.legend(loc="lower right")

    out = Path(__file__).resolve().parent / "fig4_decompose"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    print(f"[fig4] wrote {out}.pdf / .png")


if __name__ == "__main__":
    main()

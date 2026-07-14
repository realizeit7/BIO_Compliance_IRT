"""
gen_fig3_delta_theta.py — Figure 3: Δθ from system-prompt strictness.

Reads evaluator/output/phase3a_strictness_deltatheta.json and draws a grouped
bar chart of Δθ (relative to the 'none' prompt) at the 'neutral' and 'strict'
levels, for each of the three subject models, under the strict citation-
requiring judge. Output: paper/figures/fig3_delta_theta.pdf (and .png).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from _common import MODEL_SHORT, PALETTE, apply_paper_style, load_delta_theta


def main() -> None:
    apply_paper_style()
    import matplotlib.pyplot as plt

    data = load_delta_theta()
    dt = data["delta_theta_vs_none_strict"]
    models = data["models"]
    levels = ["none", "neutral", "strict"]

    x = np.arange(len(levels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, model in enumerate(models):
        vals = [dt[model][lv] for lv in levels]
        ax.bar(x + (i - 1) * width, vals, width,
               label=MODEL_SHORT.get(model, model), color=PALETTE.get(model, None))

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lv}\nprompt" for lv in levels])
    ax.set_ylabel(r"$\Delta\theta$ vs. 'none' (logits)")
    ax.set_title("System-prompt strictness effect (strict judge)")
    ax.legend(loc="upper left")

    out = Path(__file__).resolve().parent / "fig3_delta_theta"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    print(f"[fig3] wrote {out}.pdf / .png  (models={[MODEL_SHORT[m] for m in models]})")


if __name__ == "__main__":
    main()

"""
run_phase2.py — Phase 2 entry point (Rasch + classical QC).

Reads the Phase 1 responses jsonl, refits Rasch via girth, computes
point-biserial / infit / outfit per item, drops items that fail QC, and
writes the frozen healthy-item bank to evaluator/output/.

Why Rasch and not 2PL: girth's `twopl_mml` is degenerate on the 45-cell
examinee grid (every item pins to a=5.0). `twopl_jml` works but JML is
statistically inconsistent as n_items → ∞ with n_examinees fixed, which
is exactly our regime. Rasch + classical QC is the standard middle
path: keep the official model simple and identifiable, push the
broken-item detection into the diagnostics. The dormant 2PL machinery
is preserved in `evaluator/twopl.py` and `evaluator/qc.py` for a future
re-evaluation once the bank scales.

No API calls. ~1 minute on the 1,823-item × 45-examinee Phase 1 matrix.

Usage:
    source venv/bin/activate
    python3 run_phase2.py
    python3 run_phase2.py --responses logs/arena_runs/<other>/responses.jsonl
    python3 run_phase2.py --pb-threshold 0.2   # tighter discrimination QC
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from evaluator.rasch import fit_rasch
from evaluator.rasch_qc import run_rasch_qc
from evaluator.response_matrix import build_response_matrix, load_jsonl


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Phase 2: Rasch fit + classical QC.")
    parser.add_argument(
        "--responses",
        default="logs/arena_runs/phase1_baseline/responses.jsonl",
        help="Path to Phase 1 responses jsonl.",
    )
    parser.add_argument(
        "--out-prefix",
        default="evaluator/output/phase2",
        help="Output file prefix (writes <prefix>_qc_report.json and <prefix>_frozen_bank.jsonl).",
    )
    parser.add_argument(
        "--pb-threshold",
        type=float,
        default=0.0,
        help="Drop items with point-biserial <= this. 0.0 = drop only inverted/flat items.",
    )
    # Asymmetric defaults: only the upper bounds catch real misfit
    # (noisy items where the model fits poorly). Lower bounds are 0
    # because "too deterministic" items separate examinees cleanly,
    # which is what we want for a Δθ measurement bank, not a defect.
    parser.add_argument("--infit-min", type=float, default=0.0)
    parser.add_argument("--infit-max", type=float, default=1.5)
    parser.add_argument("--outfit-min", type=float, default=0.0)
    parser.add_argument("--outfit-max", type=float, default=2.0)
    args = parser.parse_args(argv)

    responses_path = Path(args.responses)
    if not responses_path.exists():
        print(f"[phase2] no responses file at {responses_path}", file=sys.stderr)
        return 1

    print(f"[phase2] reading responses ← {responses_path}")
    rm = build_response_matrix(load_jsonl(responses_path))
    print(
        f"[phase2] response matrix: {rm.n_items} items × {rm.n_examinees} examinees "
        f"(density={float((rm.matrix >= 0).mean()):.3f})"
    )

    print("[phase2] fitting Rasch via girth.rasch_mml ...")
    fit = fit_rasch(rm)
    b_finite = fit.b[np.isfinite(fit.b)]
    print(
        f"[phase2] fit done. b: median={float(np.median(b_finite)):+.3f} "
        f"min={float(b_finite.min()):+.3f} max={float(b_finite.max()):+.3f}"
    )

    print(
        f"[phase2] running QC (pb ≤ {args.pb_threshold:.2f}; "
        f"infit ∉ [{args.infit_min}, {args.infit_max}]; "
        f"outfit ∉ [{args.outfit_min}, {args.outfit_max}]) ..."
    )
    report = run_rasch_qc(
        fit,
        rm,
        pb_threshold=args.pb_threshold,
        infit_range=(args.infit_min, args.infit_max),
        outfit_range=(args.outfit_min, args.outfit_max),
    )
    s = report.summary()
    print(
        f"[phase2] QC: kept {s['n_healthy']}, dropped {s['n_dropped']} "
        f"({s['drop_rate']:.1%}); healthy median b={s['median_b_healthy']:+.3f} "
        f"pb={s['median_pb_healthy']:+.3f} infit={s['median_infit_healthy']:.3f} "
        f"outfit={s['median_outfit_healthy']:.3f}"
    )

    reason_counts: dict[str, int] = {}
    for r in report.dropped_reasons.values():
        key = r.split(" ")[0]
        reason_counts[key] = reason_counts.get(key, 0) + 1
    if reason_counts:
        print(f"[phase2] drop reasons: {reason_counts}")

    out_dir = Path(args.out_prefix).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_path = Path(f"{args.out_prefix}_qc_report.json")
    bank_path = Path(f"{args.out_prefix}_frozen_bank.jsonl")

    with qc_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": s,
                "thresholds": {
                    "pb_threshold": args.pb_threshold,
                    "infit_range": [args.infit_min, args.infit_max],
                    "outfit_range": [args.outfit_min, args.outfit_max],
                },
                "responses_source": str(responses_path),
                "n_items_input": rm.n_items,
                "n_examinees": rm.n_examinees,
                "dropped_reasons": report.dropped_reasons,
            },
            f,
            indent=2,
        )
    print(f"[phase2] wrote QC report → {qc_path}")

    with bank_path.open("w", encoding="utf-8") as f:
        for tid, b, pb, infit, outfit in zip(
            report.healthy_item_ids,
            report.healthy_b,
            report.healthy_pb,
            report.healthy_infit,
            report.healthy_outfit,
        ):
            f.write(
                json.dumps(
                    {
                        "task_id": tid,
                        "b": float(b),
                        "point_biserial": float(pb),
                        "infit": float(infit),
                        "outfit": float(outfit),
                    }
                )
                + "\n"
            )
    print(f"[phase2] wrote frozen bank → {bank_path}  ({len(report.healthy_item_ids)} items)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

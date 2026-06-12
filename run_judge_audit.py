#!/usr/bin/env python3
"""Judge Calibration Audit entry point.

Compares the project's IRT-based judge (Phase 3b strict verdicts + Rasch θ)
against DeepEval G-Eval and RAGAS metrics on the same responses.

Usage:
    python3 run_judge_audit.py --smoke          # 3 items × 1 cell × 4 metrics
    python3 run_judge_audit.py                  # 100 items × 4 zero-shot cells
    python3 run_judge_audit.py --analyze-only   # recompute report from log
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_IN", "0")

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "evaluator" / "output"
SCORES_PATH = OUT_DIR / "judge_audit_scores.jsonl"
REPORT_PATH = OUT_DIR / "judge_audit_report.json"
DISAGREEMENTS_PATH = OUT_DIR / "judge_audit_disagreements.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="3 items × 1 cell sanity check")
    parser.add_argument("--n-items", type=int, default=100)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--analyze-only", action="store_true", help="skip scoring, refit analysis")
    parser.add_argument("--grader-model", default=None, help="Groq model for G-Eval/RAGAS graders")
    args = parser.parse_args()

    from judge_audit.analysis import analyze
    from judge_audit.backends import DEFAULT_GRADER_MODEL
    from judge_audit.metrics import METRICS, run_metrics
    from judge_audit.sample import build_audit_set

    grader = args.grader_model or DEFAULT_GRADER_MODEL

    n_items = 3 if args.smoke else args.n_items
    rows = build_audit_set(n_items=n_items)
    if args.smoke:
        first_cell = rows[0].examinee_id
        rows = [r for r in rows if r.examinee_id == first_cell]
    print(f"[audit] {len(rows)} responses ({len(set(r.task_id for r in rows))} items × "
          f"{len(set(r.examinee_id for r in rows))} cells), grader={grader}")

    if not args.analyze_only:
        n_new = run_metrics(
            rows, SCORES_PATH, grader_model=grader, parallelism=args.parallelism
        )
        print(f"[audit] {n_new} new scores written → {SCORES_PATH}")

    if SCORES_PATH.exists():
        analyze(rows, SCORES_PATH, REPORT_PATH, DISAGREEMENTS_PATH, METRICS)
        print(f"[audit] report → {REPORT_PATH}")
        print(f"[audit] disagreements → {DISAGREEMENTS_PATH}")
    else:
        print("[audit] no scores yet — nothing to analyze")


if __name__ == "__main__":
    main()

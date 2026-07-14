"""Analyses over judge_audit_scores.jsonl.

1. Response-level agreement: AUC (Mann-Whitney) + point-biserial of each
   metric against the IRT judge's binary verdict.
2. Examinee-level: per-cell mean metric score vs Rasch θ (Spearman).
3. Difficulty-confounded leniency: slope of metric score on item b within
   PASS and within FAIL responses. A nonzero slope conditional on correctness
   means the metric confounds item difficulty with response quality — the
   confound IRT removes by construction.
4. Disagreement mining: top high-score-but-FAIL and low-score-but-PASS cases.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from judge_audit.sample import AuditRow


def _load_scores(scores_path: Path) -> dict[tuple[str, str, str], float]:
    scores = {}
    n_err = 0
    with open(scores_path) as f:
        for line in f:
            r = json.loads(line)
            if r["score"] is None:
                n_err += 1
                continue
            scores[(r["metric"], r["examinee_id"], r["task_id"])] = float(r["score"])
    if n_err:
        print(f"[analysis] {n_err} scoring errors excluded")
    return scores


def _auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return float("nan")
    u, _ = stats.mannwhitneyu(scores_pos, scores_neg, alternative="greater")
    return float(u / (len(scores_pos) * len(scores_neg)))


def analyze(
    rows: list[AuditRow],
    scores_path: Path,
    report_path: Path,
    disagreements_path: Path,
    metrics: tuple[str, ...],
) -> dict:
    scores = _load_scores(scores_path)
    report: dict = {"n_rows": len(rows), "metrics": {}}

    for metric in metrics:
        pairs = [
            (scores[(metric, r.examinee_id, r.task_id)], r)
            for r in rows
            if (metric, r.examinee_id, r.task_id) in scores
        ]
        if not pairs:
            print(f"[analysis] no scores for {metric}, skipping")
            continue
        s = np.array([p[0] for p in pairs])
        passed = np.array([p[1].parsed_result for p in pairs])
        b = np.array([p[1].b for p in pairs])

        m: dict = {"n_scored": len(pairs)}

        # 1. response-level agreement with IRT judge verdict
        m["auc_vs_strict_judge"] = _auc(s[passed == 1], s[passed == 0])
        if s.std() > 0 and 0 < passed.mean() < 1:
            m["point_biserial"] = float(stats.pearsonr(s, passed)[0])
        else:
            m["point_biserial"] = float("nan")

        # 2. examinee-level: mean score vs theta
        by_cell: dict[str, list[float]] = {}
        cell_theta: dict[str, float] = {}
        for sc, r in pairs:
            by_cell.setdefault(r.examinee_id, []).append(sc)
            cell_theta[r.examinee_id] = r.theta
        cells = sorted(by_cell)
        cell_means = [float(np.mean(by_cell[c])) for c in cells]
        thetas = [cell_theta[c] for c in cells]
        m["examinee_level"] = {
            "n_cells": len(cells),
            "cells": {
                c: {"mean_score": round(mu, 4), "theta": round(t, 4)}
                for c, mu, t in zip(cells, cell_means, thetas)
            },
        }
        if len(cells) >= 4:
            rho, p = stats.spearmanr(cell_means, thetas)
            m["examinee_level"]["spearman_vs_theta"] = float(rho)
            m["examinee_level"]["spearman_p"] = float(p)

        # 3. difficulty-confounded leniency: slope of score on b | verdict
        m["difficulty_slope"] = {}
        for label, mask in (("within_PASS", passed == 1), ("within_FAIL", passed == 0)):
            if mask.sum() >= 10 and s[mask].std() > 0:
                lr = stats.linregress(b[mask], s[mask])
                m["difficulty_slope"][label] = {
                    "slope": float(lr.slope),
                    "stderr": float(lr.stderr),
                    "p": float(lr.pvalue),
                    "n": int(mask.sum()),
                }

        report["metrics"][metric] = m

    # 4. disagreement mining on geval_strict (most comparable to our strict judge)
    metric = "geval_strict"
    pairs = [
        (scores[(metric, r.examinee_id, r.task_id)], r)
        for r in rows
        if (metric, r.examinee_id, r.task_id) in scores
    ]
    disagreements = []
    high_fail = sorted([p for p in pairs if p[1].parsed_result == 0], key=lambda p: -p[0])[:20]
    low_pass = sorted([p for p in pairs if p[1].parsed_result == 1], key=lambda p: p[0])[:20]
    for kind, group in (("high_geval_but_judge_FAIL", high_fail), ("low_geval_but_judge_PASS", low_pass)):
        for sc, r in group:
            disagreements.append(
                {
                    "kind": kind,
                    "geval_strict_score": sc,
                    "irt_judge_verdict": r.parsed_result,
                    "irt_judge_reason": r.judge_reason[:400],
                    "examinee_id": r.examinee_id,
                    "task_id": r.task_id,
                    "b": r.b,
                    "domain": r.domain,
                    "response": r.raw_response[:800],
                    "gold_standard": r.gold_standard[:800],
                }
            )
    with open(disagreements_path, "w") as f:
        for d in disagreements:
            f.write(json.dumps(d) + "\n")
    report["n_disagreements_dumped"] = len(disagreements)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    _print_summary(report)
    return report


def _print_summary(report: dict) -> None:
    print(f"\n{'metric':<28} {'n':>4} {'AUC':>6} {'r_pb':>6} {'rho_θ':>6} {'slope|PASS':>11} {'slope|FAIL':>11}")
    for metric, m in report["metrics"].items():
        rho = m.get("examinee_level", {}).get("spearman_vs_theta", float("nan"))
        sp = m.get("difficulty_slope", {}).get("within_PASS", {}).get("slope", float("nan"))
        sf = m.get("difficulty_slope", {}).get("within_FAIL", {}).get("slope", float("nan"))
        def fmt(x, w=6):
            return f"{x:>{w}.3f}" if isinstance(x, float) and not math.isnan(x) else " " * (w - 3) + "nan"
        print(
            f"{metric:<28} {m['n_scored']:>4} {fmt(m['auc_vs_strict_judge'])} {fmt(m['point_biserial'])} "
            f"{fmt(rho)} {fmt(sp, 11)} {fmt(sf, 11)}"
        )

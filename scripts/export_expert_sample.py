#!/usr/bin/env python3
"""Export a 50-item stratified sample from the Phase 2 frozen bank for human
expert review.

Selection design (see CLAUDE.md §9 "Human expert review"):
- 5 hard domains × 10 items each (easy_* domains excluded — trivial items
  waste expert time)
- QC filter: point_biserial ≥ 0.3, infit ≤ 1.3, outfit ≤ 1.5
- Difficulty band: b within [20th, 80th] percentile of the QC-filtered pool
  ("not too easy, not too hard")
- Verdict balance: gold standards are classified VIOLATION / COMPLIANT by
  keyword; sampling targets a 5/5 split per domain where the pool allows
- Real (FDA warning letter) items preferred: up to 3 per domain
- Deterministic: seed=42

Outputs (expert_review/):
- expert_review_sample_v1.csv   reviewer-facing form
- answer_key_v1.json            held-back analysis data (do NOT give reviewers)
"""

from __future__ import annotations

import csv
import json
import random
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANK_PATH = ROOT / "evaluator" / "output" / "phase2_frozen_bank.jsonl"
DB_PATH = ROOT / "compliance_bank.db"
OUT_DIR = ROOT / "expert_review"

SEED = 42
N_PER_DOMAIN = 10
MAX_REAL_PER_DOMAIN = 3
HARD_DOMAINS = ["21cfr11", "gcp_deviation", "gmp_deviation", "informed_consent", "promo_review"]

# QC thresholds (stricter than Phase 2 retention — we want clean items only)
MIN_PB = 0.3
MAX_INFIT = 1.3
MAX_OUTFIT = 1.5
B_PCTL_LO, B_PCTL_HI = 20, 80

_VIOLATION_PAT = re.compile(
    r"not\s+compliant|non-?compliant|violat\w+|deviation\s+from|fails?\s+to\s+(?:meet|comply)|"
    r"does\s+not\s+(?:meet|comply|satisfy)|breach",
    re.IGNORECASE,
)
_COMPLIANT_PAT = re.compile(
    r"(?:is|are|was|were|remains?|fully)\s+compliant|no\s+violation|complies\s+with|in\s+compliance",
    re.IGNORECASE,
)


def classify_verdict(gold: str) -> str:
    """VIOLATION / COMPLIANT / UNCLEAR from the gold-standard prose."""
    has_v = bool(_VIOLATION_PAT.search(gold))
    has_c = bool(_COMPLIANT_PAT.search(gold))
    if has_v and not has_c:
        return "VIOLATION"
    if has_c and not has_v:
        return "COMPLIANT"
    if has_v and has_c:
        # Both present (common: "the Red Herring is compliant, the real issue
        # is a violation"). Hard items are violation-by-design, so the
        # violation language is the operative verdict.
        return "VIOLATION"
    return "UNCLEAR"


def load_pool() -> list[dict]:
    bank = {}
    with open(BANK_PATH) as f:
        for line in f:
            row = json.loads(line)
            bank[row["task_id"]] = row

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(HARD_DOMAINS))
    rows = conn.execute(
        f"SELECT task_id, domain, context, question, gold_standard, source_type, source_ref "
        f"FROM tasks WHERE domain IN ({placeholders})",
        HARD_DOMAINS,
    ).fetchall()
    conn.close()

    pool = []
    for r in rows:
        b_row = bank.get(r["task_id"])
        if b_row is None:
            continue  # not in frozen bank
        pool.append(
            {
                "task_id": r["task_id"],
                "domain": r["domain"],
                "context": r["context"],
                "question": r["question"],
                "gold_standard": r["gold_standard"],
                "source_type": r["source_type"],
                "source_ref": r["source_ref"],
                "b": b_row["b"],
                "point_biserial": b_row["point_biserial"],
                "infit": b_row["infit"],
                "outfit": b_row["outfit"],
                "verdict": classify_verdict(r["gold_standard"]),
            }
        )
    return pool


def percentile(values: list[float], pct: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def sample_domain(items: list[dict], rng: random.Random) -> list[dict]:
    """Pick N_PER_DOMAIN items: prefer real (≤3), balance verdicts toward 5/5."""
    chosen: list[dict] = []
    remaining = list(items)
    rng.shuffle(remaining)

    def take(pred, n):
        nonlocal remaining
        picked = [x for x in remaining if pred(x)][:n]
        remaining = [x for x in remaining if x not in picked]
        chosen.extend(picked)

    # 1. Real items first (up to MAX_REAL_PER_DOMAIN, balanced verdicts)
    for verdict in ("COMPLIANT", "VIOLATION"):
        quota = MAX_REAL_PER_DOMAIN - sum(1 for c in chosen if c["source_type"] == "real")
        if quota <= 0:
            break
        take(lambda x, v=verdict: x["source_type"] == "real" and x["verdict"] == v,
             (MAX_REAL_PER_DOMAIN + 1) // 2 if verdict == "COMPLIANT" else quota)

    # 2. Fill toward 5/5 verdict balance
    while len(chosen) < N_PER_DOMAIN and remaining:
        n_v = sum(1 for c in chosen if c["verdict"] == "VIOLATION")
        n_c = sum(1 for c in chosen if c["verdict"] == "COMPLIANT")
        want = "COMPLIANT" if n_c <= n_v else "VIOLATION"
        cands = [x for x in remaining if x["verdict"] == want] or remaining
        pick = cands[0]
        remaining.remove(pick)
        chosen.append(pick)

    return chosen[:N_PER_DOMAIN]


def main() -> None:
    rng = random.Random(SEED)
    pool = load_pool()
    print(f"Joined pool (frozen bank ∩ DB, hard domains): {len(pool)} items")

    qc = [
        x for x in pool
        if x["point_biserial"] >= MIN_PB and x["infit"] <= MAX_INFIT and x["outfit"] <= MAX_OUTFIT
    ]
    print(f"After QC filter (pb≥{MIN_PB}, infit≤{MAX_INFIT}, outfit≤{MAX_OUTFIT}): {len(qc)}")

    bs = [x["b"] for x in qc]
    b_lo, b_hi = percentile(bs, B_PCTL_LO), percentile(bs, B_PCTL_HI)
    # COMPLIANT-verdict items are structurally rare (hard items are
    # violation-by-design per the High-Subtlety template), so they are exempt
    # from the difficulty band — we take any QC-passing COMPLIANT item.
    band = [
        x for x in qc
        if x["verdict"] == "COMPLIANT"
        or (b_lo <= x["b"] <= b_hi and x["verdict"] == "VIOLATION")
    ]
    n_comp = sum(1 for x in band if x["verdict"] == "COMPLIANT")
    print(f"Difficulty band b ∈ [{b_lo:.2f}, {b_hi:.2f}] (P{B_PCTL_LO}-P{B_PCTL_HI}); "
          f"pool: {len(band)} ({n_comp} COMPLIANT, band-exempt)")

    sample: list[dict] = []
    for domain in HARD_DOMAINS:
        domain_items = [x for x in band if x["domain"] == domain]
        picked = sample_domain(domain_items, rng)
        if len(picked) < N_PER_DOMAIN:
            print(f"  WARNING: {domain} pool only yielded {len(picked)}/{N_PER_DOMAIN}")
        sample.extend(picked)

    # Balance report
    print(f"\nSampled {len(sample)} items:")
    print(f"{'domain':<18} {'n':>3} {'VIOL':>5} {'COMP':>5} {'real':>5}  b range")
    for domain in HARD_DOMAINS:
        ds = [x for x in sample if x["domain"] == domain]
        n_v = sum(1 for x in ds if x["verdict"] == "VIOLATION")
        n_c = sum(1 for x in ds if x["verdict"] == "COMPLIANT")
        n_r = sum(1 for x in ds if x["source_type"] == "real")
        bs_d = [x["b"] for x in ds]
        print(f"{domain:<18} {len(ds):>3} {n_v:>5} {n_c:>5} {n_r:>5}  [{min(bs_d):+.2f}, {max(bs_d):+.2f}]")
    n_v = sum(1 for x in sample if x["verdict"] == "VIOLATION")
    n_c = sum(1 for x in sample if x["verdict"] == "COMPLIANT")
    n_r = sum(1 for x in sample if x["source_type"] == "real")
    print(f"{'TOTAL':<18} {len(sample):>3} {n_v:>5} {n_c:>5} {n_r:>5}")

    OUT_DIR.mkdir(exist_ok=True)
    rng.shuffle(sample)  # don't present grouped by domain

    csv_path = OUT_DIR / "expert_review_sample_v1.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_no", "task_id", "domain", "context", "question", "gold_standard",
                    "expert_verdict (VIOLATION/COMPLIANT)", "gold_standard_correct (Y/N)", "comments"])
        for i, x in enumerate(sample, 1):
            w.writerow([i, x["task_id"], x["domain"], x["context"], x["question"],
                        x["gold_standard"], "", "", ""])
    print(f"\nWrote {csv_path}")

    key_path = OUT_DIR / "answer_key_v1.json"
    with open(key_path, "w") as f:
        json.dump(
            {
                "seed": SEED,
                "qc_filter": {"min_pb": MIN_PB, "max_infit": MAX_INFIT, "max_outfit": MAX_OUTFIT},
                "b_band": [round(b_lo, 4), round(b_hi, 4)],
                "items": [
                    {
                        "item_no": i,
                        "task_id": x["task_id"],
                        "domain": x["domain"],
                        "derived_verdict": x["verdict"],
                        "source_type": x["source_type"],
                        "source_ref": x["source_ref"],
                        "b": x["b"],
                        "point_biserial": x["point_biserial"],
                        "infit": x["infit"],
                        "outfit": x["outfit"],
                    }
                    for i, x in enumerate(sample, 1)
                ],
            },
            f,
            indent=2,
        )
    print(f"Wrote {key_path}")


if __name__ == "__main__":
    main()

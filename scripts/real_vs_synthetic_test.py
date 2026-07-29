#!/usr/bin/env python3
"""Mann-Whitney U test: real (FDA warning letter) vs synthetic item difficulty.

Tests the descriptive real-vs-synthetic difficulty gap reported in the paper.
Reads only the committed Phase 2 frozen bank -- no API calls, no third-party
dependencies (stdlib only, so it runs in any environment).

Real items are identified by the ``TASK-REAL-`` task_id prefix (see CLAUDE.md
"Task dict schema").

Usage:
    python3 scripts/real_vs_synthetic_test.py
    python3 scripts/real_vs_synthetic_test.py --bank path/to/bank.jsonl --json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_BANK = Path("evaluator/output/phase2_frozen_bank.jsonl")
REAL_PREFIX = "TASK-REAL-"


def load_bank(path: Path) -> tuple[list[float], list[float]]:
    """Return (real_b, synthetic_b) difficulty lists from a frozen-bank jsonl."""
    real: list[float] = []
    synthetic: list[float] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            target = real if row["task_id"].startswith(REAL_PREFIX) else synthetic
            target.append(float(row["b"]))
    return real, synthetic


def _average_ranks(values: list[float]) -> tuple[list[float], float]:
    """Rank values ascending with average ranks for ties.

    Returns (ranks, tie_term) where tie_term = sum(t^3 - t) over tie groups,
    the standard correction factor for the Mann-Whitney normal approximation.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    tie_term = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        # positions i..j are tied; average rank is 1-indexed
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        t = j - i + 1
        if t > 1:
            tie_term += t ** 3 - t
        i = j + 1
    return ranks, tie_term


def mann_whitney_u(x: list[float], y: list[float]) -> dict:
    """Two-sided Mann-Whitney U test via the tie-corrected normal approximation.

    ``x`` is group 1 (here: real items), ``y`` is group 2 (synthetic).
    Reports U for both groups, the z statistic with continuity correction,
    a two-sided p-value, and rank-based effect sizes.
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        raise ValueError("both groups must be non-empty")

    ranks, tie_term = _average_ranks(x + y)
    r1 = sum(ranks[:n1])

    u1 = r1 - n1 * (n1 + 1) / 2.0   # U for group 1 (real)
    u2 = n1 * n2 - u1

    n = n1 + n2
    mu = n1 * n2 / 2.0
    # Tie-corrected standard deviation of U under H0.
    sigma = math.sqrt(
        (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
    )

    u_stat = min(u1, u2)
    # Continuity correction toward the mean.
    z = (u_stat - mu + 0.5) / sigma if sigma > 0 else 0.0
    p_two_sided = math.erfc(abs(z) / math.sqrt(2.0))

    # Effect sizes. Common-language effect size = P(random real > random synthetic).
    cles = u1 / (n1 * n2)
    cliffs_delta = 2.0 * cles - 1.0   # equals the rank-biserial correlation

    return {
        "n_real": n1,
        "n_synthetic": n2,
        "U_real": u1,
        "U_synthetic": u2,
        "U": u_stat,
        "z": z,
        "p_two_sided": p_two_sided,
        "common_language_effect_size": cles,
        "cliffs_delta": cliffs_delta,
        "tie_correction_term": tie_term,
    }


def _median(v: list[float]) -> float:
    s = sorted(v)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def summarize(real: list[float], synthetic: list[float]) -> dict:
    res = mann_whitney_u(real, synthetic)
    res.update(
        {
            "mean_b_real": sum(real) / len(real),
            "mean_b_synthetic": sum(synthetic) / len(synthetic),
            "median_b_real": _median(real),
            "median_b_synthetic": _median(synthetic),
        }
    )
    res["mean_gap"] = res["mean_b_real"] - res["mean_b_synthetic"]
    res["median_gap"] = res["median_b_real"] - res["median_b_synthetic"]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    real, synthetic = load_bank(args.bank)
    res = summarize(real, synthetic)

    if args.json:
        print(json.dumps(res, indent=2))
        return

    p = res["p_two_sided"]
    p_str = f"{p:.3g}" if p >= 1e-300 else "< 1e-300"
    print(f"Mann-Whitney U: real vs synthetic item difficulty ({args.bank})")
    print(f"  n            real={res['n_real']}  synthetic={res['n_synthetic']}")
    print(f"  mean b       real={res['mean_b_real']:+.3f}  synthetic={res['mean_b_synthetic']:+.3f}"
          f"  (gap {res['mean_gap']:+.3f})")
    print(f"  median b     real={res['median_b_real']:+.3f}  synthetic={res['median_b_synthetic']:+.3f}"
          f"  (gap {res['median_gap']:+.3f})")
    print(f"  U            {res['U']:.1f}  (U_real={res['U_real']:.1f})")
    print(f"  z            {res['z']:+.3f}")
    print(f"  p (2-sided)  {p_str}")
    print(f"  effect       Cliff's delta={res['cliffs_delta']:+.3f}  "
          f"P(real > synthetic)={res['common_language_effect_size']:.3f}")


if __name__ == "__main__":
    main()

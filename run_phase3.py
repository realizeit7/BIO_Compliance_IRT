"""
run_phase3.py — Phase 3a: Δθ from prompt-strictness on the frozen Phase 2 bank.

Pure offline analysis. For each of 10 examinee cells (3 base models ×
1 temperature × 3 strictness levels, plus the Phase 1 baseline as a
sanity check), compute MLE θ on the 1,284-item frozen Phase 2 bank using
the saved Phase 1 verdicts.

Two extra layers compared to the v1 script:

1. **Subset re-anchor.** The Phase 2 bank's `b` values were anchored
   such that the baseline examinee's MLE θ ≡ 0 on the *original*
   1,823-item set. Restricting to the 1,284-item healthy subset shifts
   the baseline's MLE θ by ~−0.7. We re-anchor inside this script:
   subtract the baseline's subset θ from every `b`, so baseline θ = 0
   by construction on the subset. Δθ comparisons within a model are
   invariant to this shift, but absolute θ values become honest.

2. **Optional lenient-judge layer.** If `--lenient-log` points at a
   responses jsonl produced by `run_phase3_rejudge_lenient.py`, the
   script computes a parallel Δθ table under the lenient judge (no
   citation requirement). The difference between strict-judge Δθ and
   lenient-judge Δθ is the format-unlock component of prompt strictness
   — i.e. how much of the apparent prompt effect is "the prompt teaches
   the model to cite at the section level" vs. "the prompt makes the
   model reason better."

Why this is offline-only:
- Phase 1 already ran every (model × temp × strictness) cell on every
  Phase 1 item. The 1,284 frozen items are a subset.
- Rasch's specific objectivity → Δθ between two examinees on the same
  item subset is invariant to the subset chosen, so the frozen bank
  is exactly the right scale to measure prompt-driven Δθ on.

Usage:
    source venv/bin/activate
    python3 run_phase3.py
    python3 run_phase3.py --temperature 0.1
    python3 run_phase3.py --lenient-log logs/arena_runs/phase3a_lenient/responses.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from girth import ability_mle

from arena.grid import _make_examinee_id

_CONFIG_PATH = Path(__file__).parent / "configs" / "phase1_baseline.yaml"


def _load_phase1_config() -> dict:
    with _CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class ExamineeKey:
    model: str
    temperature: float
    strictness: str

    @property
    def examinee_id(self) -> str:
        return _make_examinee_id(self.model, self.temperature, self.strictness)


def load_frozen_bank(path: Path) -> dict[str, float]:
    """Read frozen bank jsonl → {task_id: b}."""
    bank: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            bank[obj["task_id"]] = float(obj["b"])
    return bank


def load_responses_for_keys(
    responses_path: Path,
    keys: list[ExamineeKey],
    frozen_task_ids: set[str],
) -> dict[ExamineeKey, dict[str, int]]:
    """
    Stream responses.jsonl, keeping only rows that match one of `keys`
    AND whose task_id is in `frozen_task_ids`. Later rows for the same
    (key, task_id) overwrite earlier ones.
    """
    by_examinee_id = {k.examinee_id: k for k in keys}
    out: dict[ExamineeKey, dict[str, int]] = {k: {} for k in keys}

    with responses_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            eid = rec["examinee_id"]
            tid = rec["task_id"]
            if eid not in by_examinee_id or tid not in frozen_task_ids:
                continue
            out[by_examinee_id[eid]][tid] = int(rec["parsed_result"])
    return out


def mle_theta_one(
    responses: dict[str, int],
    bank_b: dict[str, float],
) -> tuple[float, int, float]:
    """MLE θ for one examinee against the bank. Returns (θ, n_items, P_raw)."""
    common = [tid for tid in bank_b if tid in responses]
    if not common:
        return float("nan"), 0, float("nan")
    b = np.array([bank_b[tid] for tid in common], dtype=np.float64)
    r = np.array([responses[tid] for tid in common], dtype=np.int32)
    pass_rate = float(r.mean())
    mat = r.reshape(-1, 1)
    discrimination = np.ones_like(b)
    theta = ability_mle(mat, b, discrimination)
    return float(np.atleast_1d(theta)[0]), len(common), pass_rate


def compute_table(
    by_examinee: dict[ExamineeKey, dict[str, int]],
    bank_b: dict[str, float],
    keys: list[ExamineeKey],
    baseline_key: ExamineeKey,
) -> tuple[list[dict], dict[str, float]]:
    """
    Returns (rows, anchor_info) where rows is a per-examinee list and
    anchor_info has the raw baseline θ used to re-anchor the bank.
    """
    # First pass: compute raw baseline θ on the subset under the *current*
    # bank_b (which carries the original Phase 2 anchoring).
    base_theta_raw, base_n, base_p = mle_theta_one(by_examinee[baseline_key], bank_b)

    # Re-anchor the bank: shift every b by -base_theta_raw → baseline θ = 0.
    if np.isfinite(base_theta_raw):
        bank_b_anchored = {tid: b - base_theta_raw for tid, b in bank_b.items()}
    else:
        bank_b_anchored = bank_b

    # Second pass: compute MLE θ for every examinee under the anchored bank.
    rows: list[dict] = []
    for key in keys:
        theta, n, p = mle_theta_one(by_examinee[key], bank_b_anchored)
        rows.append(
            {
                "examinee_id": key.examinee_id,
                "model": key.model,
                "temperature": key.temperature,
                "strictness": key.strictness,
                "n_items_scored": n,
                "raw_pass_rate": p,
                "theta": theta,
                "is_baseline": key == baseline_key,
            }
        )

    anchor_info = {
        "baseline_theta_raw_on_subset": base_theta_raw,
        "baseline_n_items": base_n,
        "baseline_raw_pass_rate": base_p,
        "shift_applied": -base_theta_raw if np.isfinite(base_theta_raw) else 0.0,
    }
    return rows, anchor_info


def print_table(label: str, rows: list[dict]) -> None:
    print(f"\n=== {label} ===")
    print(
        f"{'model':<26} {'temp':>5} {'strict':<8} {'n':>5} "
        f"{'P_raw':>7} {'theta':>8}"
    )
    print("-" * 70)
    for r in rows:
        tag = r["model"] + (" [BL]" if r["is_baseline"] else "")
        print(
            f"{tag:<26} {r['temperature']:>5.2f} {r['strictness']:<8} "
            f"{r['n_items_scored']:>5d} {r['raw_pass_rate']:>7.3f} {r['theta']:>+8.3f}"
        )


def delta_table(
    rows: list[dict],
    models: list[str],
    strictness_levels: list[str],
    temperature: float,
) -> dict[str, dict[str, float]]:
    deltas: dict[str, dict[str, float]] = {}
    for m in models:
        per_strict = {
            r["strictness"]: r["theta"]
            for r in rows
            if r["model"] == m and abs(r["temperature"] - temperature) < 1e-9
        }
        ref = per_strict.get("none", float("nan"))
        deltas[m] = {s: per_strict.get(s, float("nan")) - ref for s in strictness_levels}
    return deltas


def print_delta_table(label: str, deltas: dict[str, dict[str, float]], strictness_levels: list[str]) -> None:
    print(f"\n=== {label} (Δθ vs same-model strictness=none) ===")
    header = f"{'model':<26}" + "".join(f"{s:>9}" for s in strictness_levels)
    print(header)
    print("-" * len(header))
    for model, d in deltas.items():
        line = f"{model:<26}" + "".join(f"{d[s]:>+9.3f}" for s in strictness_levels)
        print(line)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3a: prompt-strictness Δθ on the frozen Phase 2 bank."
    )
    parser.add_argument(
        "--bank",
        default="evaluator/output/phase2_frozen_bank.jsonl",
        help="Frozen Phase 2 bank jsonl (anchored b per task_id).",
    )
    parser.add_argument(
        "--responses",
        default="logs/arena_runs/phase1_baseline/responses.jsonl",
        help="Phase 1 responses jsonl (strict-judge verdicts).",
    )
    parser.add_argument(
        "--lenient-log",
        default="logs/arena_runs/phase3a_lenient/responses.jsonl",
        help="Optional lenient-judge log from run_phase3_rejudge_lenient.py. "
             "If present, the script computes a parallel lenient Δθ table.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Temperature of the subject grid. Phase 3a default: 0.4.",
    )
    parser.add_argument(
        "--out",
        default="evaluator/output/phase3a_strictness_deltatheta.json",
        help="Output JSON for the Δθ tables.",
    )
    args = parser.parse_args(argv)

    bank_path = Path(args.bank)
    if not bank_path.exists():
        print(f"[phase3a] missing frozen bank at {bank_path}", file=sys.stderr)
        return 1
    responses_path = Path(args.responses)
    if not responses_path.exists():
        print(f"[phase3a] missing responses at {responses_path}", file=sys.stderr)
        return 1

    bank_b = load_frozen_bank(bank_path)
    print(f"[phase3a] frozen bank: {len(bank_b)} items  ← {bank_path}")

    cfg = _load_phase1_config()
    models: list[str] = cfg["grid"]["models"]
    strictness_levels: list[str] = cfg["grid"]["strictness_levels"]
    _baseline_cfg = cfg["baseline"]
    baseline_key = ExamineeKey(
        model=_baseline_cfg["model"],
        temperature=float(_baseline_cfg["temperature"]),
        strictness=_baseline_cfg["strictness"],
    )

    subject_keys = [
        ExamineeKey(model=m, temperature=args.temperature, strictness=s)
        for m in models
        for s in strictness_levels
    ]
    all_keys = subject_keys + [baseline_key]

    print(
        f"[phase3a] examinees: {len(subject_keys)} subject cells "
        f"({len(models)} models × t={args.temperature} × {len(strictness_levels)} strictness) "
        f"+ 1 baseline ({baseline_key.model} @ t={baseline_key.temperature}, {baseline_key.strictness})"
        f"  [config: {_CONFIG_PATH}]"
    )

    # --- Strict judge pass --------------------------------------------------
    print(f"\n[phase3a] strict-judge pass ← {responses_path}")
    by_strict = load_responses_for_keys(responses_path, all_keys, set(bank_b))
    rows_strict, anchor_strict = compute_table(by_strict, bank_b, all_keys, baseline_key)
    print(
        f"[phase3a] subset re-anchor: baseline raw θ on 1,284 items = "
        f"{anchor_strict['baseline_theta_raw_on_subset']:+.4f} → "
        f"shift b by {anchor_strict['shift_applied']:+.4f} so baseline θ ≡ 0."
    )
    print_table("STRICT JUDGE (citation required)", rows_strict)
    deltas_strict = delta_table(rows_strict, models, strictness_levels, args.temperature)
    print_delta_table("STRICT JUDGE", deltas_strict, strictness_levels)

    # --- Optional lenient judge pass ---------------------------------------
    rows_lenient: list[dict] | None = None
    deltas_lenient: dict[str, dict[str, float]] | None = None
    deltas_format: dict[str, dict[str, float]] | None = None
    anchor_lenient: dict | None = None

    lenient_path = Path(args.lenient_log)
    if lenient_path.exists():
        print(f"\n[phase3a] lenient-judge pass ← {lenient_path}")
        by_lenient = load_responses_for_keys(lenient_path, all_keys, set(bank_b))
        # Heuristic: if the lenient log doesn't include the baseline (it
        # wasn't in the rejudge scope), use the strict baseline θ for
        # anchoring so the two scales are comparable.
        if not by_lenient[baseline_key]:
            by_lenient[baseline_key] = by_strict[baseline_key]
            print(
                "[phase3a] lenient log lacked baseline rows; reused strict baseline "
                "verdicts for the lenient anchor (baseline solver output is identical)."
            )
        rows_lenient, anchor_lenient = compute_table(
            by_lenient, bank_b, all_keys, baseline_key
        )
        print(
            f"[phase3a] subset re-anchor (lenient): baseline raw θ = "
            f"{anchor_lenient['baseline_theta_raw_on_subset']:+.4f} → "
            f"shift b by {anchor_lenient['shift_applied']:+.4f}."
        )
        print_table("LENIENT JUDGE (no citation required)", rows_lenient)
        deltas_lenient = delta_table(rows_lenient, models, strictness_levels, args.temperature)
        print_delta_table("LENIENT JUDGE", deltas_lenient, strictness_levels)

        # Format-unlock = strict Δθ − lenient Δθ. Conceptually: how much of
        # the apparent prompt effect was buying the model the citation
        # format the strict judge requires, vs. genuinely improving its
        # compliance reasoning.
        deltas_format = {
            m: {
                s: deltas_strict[m][s] - deltas_lenient[m][s]
                for s in strictness_levels
            }
            for m in models
        }
        print_delta_table(
            "FORMAT-UNLOCK COMPONENT (strict Δθ − lenient Δθ)",
            deltas_format,
            strictness_levels,
        )
        print(
            "\nInterpretation: positive entries = the strict prompt's effect under the "
            "strict judge was inflated by 'satisfy citation format'. Lenient Δθ = "
            "the residual reasoning improvement, the more honest 'prompt engineering' number."
        )
    else:
        print(
            f"\n[phase3a] lenient log not found at {lenient_path}; "
            "skipping format-unlock analysis. Run run_phase3_rejudge_lenient.py first."
        )

    # --- Persist ------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "frozen_bank": str(bank_path),
        "responses_source_strict": str(responses_path),
        "responses_source_lenient": str(lenient_path) if lenient_path.exists() else None,
        "n_frozen_items": len(bank_b),
        "subject_temperature": args.temperature,
        "strictness_levels": strictness_levels,
        "models": models,
        "subset_anchor_strict": anchor_strict,
        "subset_anchor_lenient": anchor_lenient,
        "rows_strict": rows_strict,
        "rows_lenient": rows_lenient,
        "delta_theta_vs_none_strict": deltas_strict,
        "delta_theta_vs_none_lenient": deltas_lenient,
        "format_unlock_delta": deltas_format,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[phase3a] wrote Δθ report → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

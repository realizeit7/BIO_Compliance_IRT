"""
run_phase3b.py — Phase 3b: Δθ from agent architecture on the frozen Phase 2 bank.

Runs four agent variants (zero_shot, retrieval, critic, step_decomp) against
the full 1,284-item frozen Phase 2 bank, fits Rasch MLE θ per examinee cell,
and writes per-agent-type Δθ relative to the zero_shot baseline.

Usage:
    source venv/bin/activate
    python3 run_phase3b.py                  # full run
    python3 run_phase3b.py --smoke          # 3 items × 16 cells (sanity check)
    python3 run_phase3b.py --calibrate-only # skip arena, refit IRT on existing log
    python3 run_phase3b.py --config configs/phase3b_variants.yaml

Grid (default): 2 models × 1 temp × 2 strictness × 4 agent_types = 16 cells.
Budget: ~41k solver calls + ~20k judge calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from girth import ability_mle

from agents.base import Agent, Item
from agents.critic import CriticAgent
from agents.retrieval import RetrievalAgent
from agents.step_decomp import StepDecompositionAgent
from agents.zero_shot import ZeroShotAgent
from arena.grid import ExamineeConfig, _make_examinee_id, build_grid
from arena.runner import ArenaRunner
from harness.cfr_store import CFRStore
from harness.groq_client import GroqClient
from harness.judge import Judge

_REPO_ROOT = Path(__file__).parent
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "phase3b_variants.yaml"


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_frozen_bank(path: Path) -> dict[str, float]:
    bank: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                bank[obj["task_id"]] = float(obj["b"])
    return bank


def _load_frozen_items(bank_path: Path) -> list[Item]:
    """Load only the items present in the frozen bank from the DB."""
    from arena.loader import ItemLoader

    bank_ids: set[str] = set()
    with bank_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                bank_ids.add(json.loads(line)["task_id"])

    loader = ItemLoader()
    all_items = loader.fetch_all()
    loader.close()
    items = [it for it in all_items if it.task_id in bank_ids]
    missing = len(bank_ids) - len(items)
    if missing:
        print(f"[phase3b] warn: {missing} bank items not found in DB (expected if real items were deleted)")
    return items


def _mle_theta(responses: dict[str, int], bank_b: dict[str, float]) -> tuple[float, int]:
    common = [tid for tid in bank_b if tid in responses]
    if not common:
        return float("nan"), 0
    b = np.array([bank_b[tid] for tid in common], dtype=np.float64)
    r = np.array([responses[tid] for tid in common], dtype=np.int32).reshape(-1, 1)
    disc = np.ones_like(b)
    theta = ability_mle(r, b, disc)
    return float(np.atleast_1d(theta)[0]), len(common)


def _collect_responses(
    log_path: Path,
    examinee_ids: set[str],
    frozen_task_ids: set[str],
) -> dict[str, dict[str, int]]:
    """Stream responses.jsonl → {examinee_id: {task_id: parsed_result}}."""
    out: dict[str, dict[str, int]] = {eid: {} for eid in examinee_ids}
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            eid = rec["examinee_id"]
            tid = rec["task_id"]
            if eid in out and tid in frozen_task_ids:
                out[eid][tid] = int(rec["parsed_result"])
    return out


def _build_agent_factory(
    cfg: dict[str, Any],
    client: GroqClient,
    cfr_store: CFRStore,
) -> Any:
    retrieval_k: int = cfg.get("retrieval", {}).get("k", 3)
    critic_rounds: int = cfg.get("critic", {}).get("n_rounds", 1)

    def factory(examinee: ExamineeConfig) -> Agent:
        common = dict(
            examinee_id=examinee.examinee_id,
            model=examinee.model,
            temperature=examinee.temperature,
            strictness=examinee.strictness,
            client=client,
            seed=examinee.seed,
        )
        atype = examinee.agent_type
        if atype == "zero_shot":
            return ZeroShotAgent(**common)
        if atype == "retrieval":
            return RetrievalAgent(**common, cfr_store=cfr_store, k=retrieval_k)
        if atype == "critic":
            return CriticAgent(**common, n_rounds=critic_rounds)
        if atype == "step_decomp":
            return StepDecompositionAgent(**common)
        raise ValueError(f"Unknown agent_type: {atype!r}")

    return factory


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Phase 3b: agent-variant Δθ on frozen bank.")
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true", help="3 items × 16 cells sanity check")
    parser.add_argument("--calibrate-only", action="store_true", help="Skip arena; refit IRT")
    parser.add_argument("--out", default="evaluator/output/phase3b_agent_deltatheta.json")
    parser.add_argument(
        "--phase1-log",
        default="logs/arena_runs/phase1_baseline/responses.jsonl",
        help="Phase 1 responses jsonl used to load baseline examinee responses for anchoring. "
             "Only needed if the baseline examinee is not in the Phase 3b grid (the default).",
    )
    args = parser.parse_args(argv)

    cfg = _load_config(Path(args.config))
    run_id: str = cfg["run_id"]
    log_dir = _REPO_ROOT / cfg.get("log_dir", "logs/arena_runs")
    bank_path = _REPO_ROOT / cfg["bank_path"]

    if not bank_path.exists():
        print(f"[phase3b] frozen bank not found: {bank_path}", file=sys.stderr)
        return 1

    bank_b = _load_frozen_bank(bank_path)
    print(f"[phase3b] frozen bank: {len(bank_b)} items")

    # Build examinee grid
    gcfg = cfg["grid"]
    grid = build_grid(
        models=gcfg["models"],
        temperatures=gcfg["temperatures"],
        strictness_levels=gcfg["strictness_levels"],
        agent_types=gcfg.get("agent_types", ["zero_shot"]),
        seed=gcfg.get("seed"),
    )
    print(f"[phase3b] grid: {len(grid)} examinee cells")

    # Baseline key (for re-anchoring)
    bl = cfg["baseline"]
    baseline_id = _make_examinee_id(
        bl["model"], float(bl["temperature"]), bl["strictness"],
        bl.get("agent_type", "zero_shot"),
    )

    if not args.calibrate_only:
        # Load items
        items = _load_frozen_items(bank_path)
        if args.smoke:
            items = items[:3]
            print(f"[phase3b] smoke mode: {len(items)} items")
        else:
            print(f"[phase3b] items to administer: {len(items)}")

        # Shared harness objects
        api_key = None  # GroqClient reads GROQ_API_KEY from env
        client = GroqClient()
        judge_strict = Judge(client=client, strict=True)
        judge_lenient = Judge(client=client, strict=False)

        # CFR store (shared, lazy-loaded)
        cfr_store = CFRStore()

        agent_factory = _build_agent_factory(cfg, client, cfr_store)

        runner = ArenaRunner(
            run_id=run_id,
            log_dir=log_dir,
            client=client,
            judge=judge_strict,
            judge_lenient=judge_lenient,
            parallelism=int(cfg.get("parallelism", 8)),
            agent_factory=agent_factory,
        )
        n_written = runner.run(examinees=grid, items=items)
        print(f"[phase3b] arena complete: {n_written} new entries written")

    # IRT fitting
    log_path = log_dir / run_id / "responses.jsonl"
    if not log_path.exists():
        print(f"[phase3b] no log found at {log_path} — run without --calibrate-only first", file=sys.stderr)
        return 1

    examinee_ids = {e.examinee_id for e in grid}
    frozen_task_ids = set(bank_b)
    by_examinee = _collect_responses(log_path, examinee_ids, frozen_task_ids)

    # Compute raw θ for baseline (to re-anchor onto the Phase 1 scale where baseline θ=0).
    # The baseline examinee (t=0.1) is not in the Phase 3b grid (t=0.4), so load its
    # responses from the Phase 1 log if available.
    baseline_responses = by_examinee.get(baseline_id, {})
    if not baseline_responses:
        phase1_log = _REPO_ROOT / args.phase1_log
        if phase1_log.exists():
            print(f"[phase3b] baseline not in grid; loading from Phase 1 log: {phase1_log}")
            phase1_data = _collect_responses(phase1_log, {baseline_id}, frozen_task_ids)
            baseline_responses = phase1_data.get(baseline_id, {})
        if not baseline_responses:
            print(f"[phase3b] warn: baseline {baseline_id!r} has no responses; anchoring skipped")
    if baseline_responses:
        baseline_theta_raw, _ = _mle_theta(baseline_responses, bank_b)
        anchor_shift = -baseline_theta_raw if np.isfinite(baseline_theta_raw) else 0.0
    else:
        baseline_theta_raw = float("nan")
        anchor_shift = 0.0
    bank_b_anchored = {tid: b + anchor_shift for tid, b in bank_b.items()}

    print(f"\n[phase3b] baseline raw θ = {baseline_theta_raw:+.4f} → anchor shift = {anchor_shift:+.4f}")

    # Fit θ for every cell
    rows: list[dict] = []
    for examinee in grid:
        resps = by_examinee.get(examinee.examinee_id, {})
        theta, n = _mle_theta(resps, bank_b_anchored)
        rows.append({
            "examinee_id": examinee.examinee_id,
            "model": examinee.model,
            "temperature": examinee.temperature,
            "strictness": examinee.strictness,
            "agent_type": examinee.agent_type,
            "n_items_scored": n,
            "raw_pass_rate": float(np.mean(list(resps.values()))) if resps else float("nan"),
            "theta": theta,
            "is_baseline": examinee.examinee_id == baseline_id,
        })

    # Δθ vs zero_shot within (model, strictness)
    delta_rows: list[dict] = []
    for row in rows:
        if row["agent_type"] == "zero_shot":
            continue
        ref_theta = next(
            (r["theta"] for r in rows
             if r["model"] == row["model"]
             and abs(r["temperature"] - row["temperature"]) < 1e-9
             and r["strictness"] == row["strictness"]
             and r["agent_type"] == "zero_shot"),
            float("nan"),
        )
        delta_rows.append({
            **row,
            "delta_theta_vs_zero_shot": row["theta"] - ref_theta,
        })

    # Print summary
    print(f"\n{'model':<26} {'strict':<8} {'agent_type':<14} {'n':>5} {'P':>6} {'θ':>8} {'Δθ':>8}")
    print("-" * 82)
    for row in rows:
        drow = next((d for d in delta_rows if d["examinee_id"] == row["examinee_id"]), None)
        delta_str = f"{drow['delta_theta_vs_zero_shot']:>+8.3f}" if drow else "    ref"
        print(
            f"{row['model']:<26} {row['strictness']:<8} {row['agent_type']:<14} "
            f"{row['n_items_scored']:>5} {row['raw_pass_rate']:>6.3f} "
            f"{row['theta']:>+8.3f} {delta_str}"
        )

    # Persist
    out_path = _REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "frozen_bank": str(bank_path),
        "n_frozen_items": len(bank_b),
        "baseline_examinee_id": baseline_id,
        "baseline_theta_raw": baseline_theta_raw,
        "anchor_shift": anchor_shift,
        "grid_config": gcfg,
        "rows": rows,
        "delta_rows": delta_rows,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[phase3b] wrote Δθ report → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

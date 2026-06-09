"""
run_phase1.py — Phase 1 entry point.

Reads configs/phase1_baseline.yaml, builds the harness + grid + items,
drives the Arena runner, then fits 1PL Rasch via /evaluator/ and anchors
the baseline examinee at θ = 0.

This is a thin glue script — all real logic lives under /harness/,
/agents/, /arena/, /evaluator/.

Usage:
    source venv/bin/activate && export GROQ_API_KEY=<your_key>
    python3 run_phase1.py                       # full Phase 1 from config
    python3 run_phase1.py --smoke               # 3 items × 3 examinees, sanity check
    python3 run_phase1.py --config <path>       # alternate config
    python3 run_phase1.py --calibrate-only      # skip arena, just fit IRT on existing log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from agents.zero_shot import STRICTNESS_LEVELS  # noqa: F401  (validates strictness names)
from arena.grid import ExamineeConfig, build_grid
from arena.loader import ItemLoader
from arena.runner import ArenaRunner
from evaluator.rasch import anchor_baseline_to_zero, fit_rasch
from evaluator.response_matrix import build_response_matrix, load_jsonl
from harness.groq_client import GroqClient
from harness.judge import Judge


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def baseline_examinee_id(grid: list[ExamineeConfig], baseline_cfg: dict) -> str | None:
    for e in grid:
        if (
            e.model == baseline_cfg["model"]
            and abs(e.temperature - float(baseline_cfg["temperature"])) < 1e-9
            and e.strictness == baseline_cfg["strictness"]
        ):
            return e.examinee_id
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Phase 1: 1PL baseline calibration.")
    parser.add_argument("--config", default="configs/phase1_baseline.yaml")
    parser.add_argument("--smoke", action="store_true", help="3 items × 3 examinees only.")
    parser.add_argument(
        "--calibrate-only",
        action="store_true",
        help="Skip Arena run; load existing jsonl and fit IRT.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run_id = cfg["run_id"] + ("_smoke" if args.smoke else "")
    log_dir = Path(cfg["log_dir"])

    # --- Build items + grid -------------------------------------------------
    loader = ItemLoader(cfg["items"]["db_path"])
    if args.smoke:
        items = loader.stratified_sample(
            sample_size=3, seed=int(cfg["items"]["sample_seed"])
        )
    elif cfg["items"].get("sample_size"):
        items = loader.stratified_sample(
            sample_size=int(cfg["items"]["sample_size"]),
            seed=int(cfg["items"]["sample_seed"]),
            source_types=tuple(cfg["items"]["source_types"]),
        )
    else:
        items = loader.fetch_all(source_types=tuple(cfg["items"]["source_types"]))

    grid = build_grid(
        models=list(cfg["grid"]["models"]),
        temperatures=[float(t) for t in cfg["grid"]["temperatures"]],
        strictness_levels=list(cfg["grid"]["strictness_levels"]),
        seed=int(cfg["grid"]["seed"]) if cfg["grid"].get("seed") is not None else None,
    )
    # In smoke mode keep the full grid (so the baseline examinee is exercised)
    # but cap items at 3 — so total attempts ~= |grid| * 3.

    print(f"[phase1] run_id={run_id}  items={len(items)}  examinees={len(grid)}")
    print(f"[phase1] expected attempts = {len(items) * len(grid)}")

    # --- Run the Arena ------------------------------------------------------
    if not args.calibrate_only:
        client = GroqClient(
            timeout=float(cfg["runtime"]["request_timeout_seconds"]),
            max_retries=int(cfg["runtime"]["max_retries"]),
        )
        judge_strict = Judge(
            client, model=cfg["judge"]["model"], strict=bool(cfg["judge"]["strict"])
        )
        judge_lenient = (
            Judge(client, model=cfg["judge"]["model"], strict=False)
            if cfg["judge"].get("use_lenient_for_easy_domains", True)
            else judge_strict
        )

        runner = ArenaRunner(
            run_id=run_id,
            log_dir=log_dir,
            client=client,
            judge=judge_strict,
            judge_lenient=judge_lenient,
            parallelism=int(cfg["runtime"]["parallelism"]),
        )
        runner.run(examinees=grid, items=items)

    # --- Offline IRT calibration -------------------------------------------
    log_path = log_dir / run_id / "responses.jsonl"
    if not log_path.exists():
        print(f"[phase1] no log file at {log_path}; nothing to calibrate.")
        return 1

    rm = build_response_matrix(load_jsonl(log_path))
    print(
        f"[phase1] response matrix: {rm.n_items} items × {rm.n_examinees} examinees "
        f"(density={float((rm.matrix >= 0).mean()):.3f})"
    )

    fit = fit_rasch(rm)

    baseline_id = baseline_examinee_id(grid, cfg["baseline"])
    if baseline_id is None:
        print(
            "[phase1] baseline (model, temperature, strictness) not in configured grid; "
            "b values unanchored."
        )
    elif baseline_id not in rm.examinee_ids:
        print(f"[phase1] baseline {baseline_id} not yet in log; b values unanchored.")
    else:
        anchored = anchor_baseline_to_zero(fit, rm, baseline_id)
        if anchored.anchored:
            print(
                f"[phase1] baseline {baseline_id} raw θ = {anchored.theta_baseline_raw:+.3f} "
                f"→ anchored to 0; b shifted by {-anchored.theta_baseline_raw:+.3f}"
            )
            fit = anchored
        else:
            print(
                f"[phase1] baseline {baseline_id} produced non-finite MLE θ "
                f"({anchored.theta_baseline_raw}); keeping unanchored b values."
            )

    # --- Persist b estimates -----------------------------------------------
    out_dir = Path("evaluator/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}_phase1_b.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for tid, b in zip(fit.item_ids, fit.b):
            f.write(
                json.dumps(
                    {
                        "task_id": tid,
                        "b": float(b),
                        "anchored": fit.anchored,
                        "baseline_examinee_id": fit.baseline_examinee_id,
                    }
                )
                + "\n"
            )
    print(f"[phase1] wrote {len(fit.item_ids)} b estimates → {out_path}")
    print(
        f"[phase1] b summary: min={float(fit.b.min()):+.3f} "
        f"median={float(__import__('numpy').median(fit.b)):+.3f} "
        f"max={float(fit.b.max()):+.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""
run_phase3_rejudge_lenient.py — Re-grade Phase 1 responses with the LENIENT judge.

Phase 3a's strict-judge Δθ on the 70b confounds two effects:
  (a) the prompt teaches the model to *cite CFR sections at section level*
      (the strict judge's hard requirement), and
  (b) the prompt makes the model reason better about compliance.

To disentangle these, we re-grade the same `raw_response` text the model
already produced in Phase 1 with the *lenient* judge, which scores only
the conclusion + reasoning and ignores citation format. The
"format-unlock" component of Δθ is then:

    Δθ_format = Δθ_under_strict_judge − Δθ_under_lenient_judge

This script does NOT re-run any solver — it reuses the saved
`raw_response` text from Phase 1's responses.jsonl. Only judge calls hit
the API.

Scope: 9 subject cells (3 models × 1 temperature × 3 strictness levels)
× 1,284 frozen Phase 2 items = 11,556 judge calls. Resumable: pairs
already in the output log are skipped, so an interrupted run can be
restarted.

Usage:
    source venv/bin/activate && export $(cat .env | xargs)
    python3 run_phase3_rejudge_lenient.py
    python3 run_phase3_rejudge_lenient.py --temperature 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from arena.grid import _make_examinee_id
from arena.loader import ItemLoader
from arena.schema import ArenaLogEntry
from harness.groq_client import GroqClient
from harness.judge import Judge


def collect_phase1_rows(
    responses_path: Path,
    target_examinee_ids: set[str],
    frozen_task_ids: set[str],
) -> dict[tuple[str, str], dict]:
    """
    Stream Phase 1 responses jsonl and return the most-recent row per
    (examinee_id, task_id) for the targeted subset.
    """
    rows: dict[tuple[str, str], dict] = {}
    with responses_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            eid, tid = rec["examinee_id"], rec["task_id"]
            if eid in target_examinee_ids and tid in frozen_task_ids:
                rows[(eid, tid)] = rec
    return rows


def already_done(log_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not log_path.exists():
        return done
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add((obj["examinee_id"], obj["task_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Re-grade Phase 1 responses with the lenient judge for Phase 3a."
    )
    parser.add_argument(
        "--bank",
        default="evaluator/output/phase2_frozen_bank.jsonl",
        help="Frozen Phase 2 bank (defines the 1,284 items in scope).",
    )
    parser.add_argument(
        "--responses",
        default="logs/arena_runs/phase1_baseline/responses.jsonl",
        help="Phase 1 responses jsonl (source of saved raw_response text).",
    )
    parser.add_argument(
        "--db",
        default="compliance_bank.db",
        help="SQLite bank for question + gold_standard lookup.",
    )
    parser.add_argument(
        "--out-run-id",
        default="phase3a_lenient",
        help="Run ID for the lenient-judged log (writes logs/arena_runs/<id>/responses.jsonl).",
    )
    parser.add_argument("--log-dir", default="logs/arena_runs")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Subject grid temperature to re-grade.",
    )
    parser.add_argument(
        "--judge-model",
        default="llama-3.3-70b-versatile",
        help="LLM used as the lenient judge (matches Phase 1 default).",
    )
    parser.add_argument("--parallelism", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args(argv)

    bank_path = Path(args.bank)
    responses_path = Path(args.responses)
    if not bank_path.exists():
        print(f"[rejudge] missing frozen bank at {bank_path}", file=sys.stderr)
        return 1
    if not responses_path.exists():
        print(f"[rejudge] missing responses at {responses_path}", file=sys.stderr)
        return 1

    # 1,284 frozen task_ids
    frozen_task_ids: set[str] = set()
    with bank_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frozen_task_ids.add(json.loads(line)["task_id"])
    print(f"[rejudge] frozen bank: {len(frozen_task_ids)} task_ids")

    # 9 subject cells × t=temperature
    models = [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
    ]
    strictness_levels = ["none", "neutral", "strict"]
    subject_examinee_ids = {
        _make_examinee_id(m, args.temperature, s)
        for m in models
        for s in strictness_levels
    }
    print(
        f"[rejudge] subject cells: {len(subject_examinee_ids)} examinees "
        f"({len(models)} models × t={args.temperature} × {len(strictness_levels)} strictness)"
    )

    # Load Items (need question + gold_standard for the judge prompt)
    loader = ItemLoader(args.db)
    all_items = {it.task_id: it for it in loader.fetch_all()}
    loader.close()
    print(f"[rejudge] loaded {len(all_items)} items from {args.db}")

    # Collect the rows we need to re-judge
    print(f"[rejudge] streaming Phase 1 responses ← {responses_path}")
    target_rows = collect_phase1_rows(
        responses_path, subject_examinee_ids, frozen_task_ids
    )
    print(f"[rejudge] {len(target_rows)} (examinee × item) pairs in scope")

    # Resumability
    run_dir = Path(args.log_dir) / args.out_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "responses.jsonl"
    done = already_done(log_path)
    print(f"[rejudge] log file: {log_path}  (already_done={len(done)})")

    todo: list[dict] = [
        rec for (eid, tid), rec in target_rows.items() if (eid, tid) not in done
    ]
    if not todo:
        print("[rejudge] nothing to do — all pairs already re-judged.")
        return 0
    print(f"[rejudge] pairs to re-judge: {len(todo)}")

    # Build judge
    client = GroqClient(timeout=args.timeout, max_retries=args.max_retries)
    judge = Judge(client, model=args.judge_model, strict=False)

    write_lock = threading.Lock()

    def append(entry: ArenaLogEntry) -> None:
        line = entry.model_dump_json() + "\n"
        with write_lock:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line)

    def regrade_one(rec: dict) -> ArenaLogEntry | None:
        tid = rec["task_id"]
        eid = rec["examinee_id"]
        item = all_items.get(tid)
        if item is None:
            # Item disappeared from DB since Phase 1 — skip rather than bias.
            return None
        try:
            raw_response = rec.get("raw_response", "") or ""
            verdict = judge.grade(
                question=item.question,
                gold_standard=item.gold_standard,
                response=raw_response,
            )
            entry = ArenaLogEntry(
                run_id=args.out_run_id,
                examinee_id=eid,
                task_id=tid,
                agent_class=rec.get("agent_class", "ZeroShotAgent"),
                model=rec["model"],
                temperature=rec["temperature"],
                strictness=rec.get("strictness"),
                seed=rec.get("seed"),
                domain=rec.get("domain", item.domain),
                source_type=rec.get("source_type", item.source_type),
                raw_response=raw_response,
                retry_count=rec.get("retry_count", 0),
                finish_reason=rec.get("finish_reason"),
                latency_seconds=rec.get("latency_seconds", 0.0),
                solver_error_type=rec.get("solver_error_type"),
                solver_error_message=rec.get("solver_error_message"),
                judge_model=judge.model,
                judge_strict=False,
                judge_raw=verdict.raw_text,
                judge_reason=verdict.reason,
                judge_error_type=verdict.error_type,
                parsed_result=1 if verdict.passed else 0,
                extra={"rejudged_from": "phase1_baseline", "judge_pass": "lenient"},
            )
            return entry
        except Exception as exc:  # noqa: BLE001 — Rule 3
            tb = traceback.format_exc()
            return ArenaLogEntry(
                run_id=args.out_run_id,
                examinee_id=eid,
                task_id=tid,
                agent_class=rec.get("agent_class", "ZeroShotAgent"),
                model=rec["model"],
                temperature=rec["temperature"],
                strictness=rec.get("strictness"),
                seed=rec.get("seed"),
                domain=rec.get("domain", item.domain),
                source_type=rec.get("source_type", item.source_type),
                raw_response=rec.get("raw_response", ""),
                retry_count=0,
                finish_reason=None,
                latency_seconds=0.0,
                solver_error_type=None,
                solver_error_message=None,
                judge_model=judge.model,
                judge_strict=False,
                judge_raw="",
                judge_reason=f"Rejudge-level exception: {exc}",
                judge_error_type="RejudgeException",
                parsed_result=0,
                extra={"traceback": tb, "judge_pass": "lenient"},
            )

    n_written = 0
    with ThreadPoolExecutor(max_workers=args.parallelism) as pool:
        futures = [pool.submit(regrade_one, rec) for rec in todo]
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="lenient-rejudge", unit="pair"
        ):
            entry = fut.result()
            if entry is not None:
                append(entry)
                n_written += 1
    print(f"[rejudge] wrote {n_written} new lenient verdicts → {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

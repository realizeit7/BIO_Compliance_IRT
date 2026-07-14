"""Build the audit set: phase3b responses ∩ frozen bank ∩ DB gold standards.

Default selection: 100 frozen-bank items stratified by difficulty quartile
(25 per quartile, seed=42) × the 4 zero-shot cells (2 models × 2 strictness).
Rows with solver errors or empty responses are excluded.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANK_PATH = ROOT / "evaluator" / "output" / "phase2_frozen_bank.jsonl"
RESPONSES_PATH = ROOT / "logs" / "arena_runs" / "phase3b_variants" / "responses.jsonl"
THETA_PATH = ROOT / "evaluator" / "output" / "phase3b_agent_deltatheta.json"
DB_PATH = ROOT / "compliance_bank.db"

SEED = 42


@dataclass
class AuditRow:
    examinee_id: str
    task_id: str
    model: str
    strictness: str
    agent_type: str
    raw_response: str
    parsed_result: int  # IRT judge verdict: 1=PASS, 0=FAIL
    judge_reason: str
    b: float
    theta: float
    domain: str
    context: str
    question: str
    gold_standard: str


def _load_bank() -> dict[str, float]:
    bank = {}
    with open(BANK_PATH) as f:
        for line in f:
            row = json.loads(line)
            bank[row["task_id"]] = row["b"]
    return bank


def _load_theta() -> dict[str, float]:
    data = json.loads(THETA_PATH.read_text())
    return {r["examinee_id"]: r["theta"] for r in data["rows"]}


def _load_items() -> dict[str, dict]:
    import sqlite3

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT task_id, domain, context, question, gold_standard FROM tasks"
    ).fetchall()
    conn.close()
    return {r["task_id"]: dict(r) for r in rows}


def select_items(bank: dict[str, float], task_ids: set[str], n_items: int, seed: int = SEED) -> list[str]:
    """Stratified by difficulty quartile of b over the available items."""
    avail = sorted(t for t in task_ids if t in bank)
    if n_items >= len(avail):
        return avail
    avail.sort(key=lambda t: bank[t])
    rng = random.Random(seed)
    quartiles = [avail[i * len(avail) // 4 : (i + 1) * len(avail) // 4] for i in range(4)]
    per_q = n_items // 4
    chosen: list[str] = []
    for q in quartiles:
        chosen.extend(rng.sample(q, min(per_q, len(q))))
    # top up if n_items not divisible by 4
    leftovers = [t for t in avail if t not in set(chosen)]
    while len(chosen) < n_items and leftovers:
        chosen.append(leftovers.pop(rng.randrange(len(leftovers))))
    return chosen


def build_audit_set(
    *,
    n_items: int = 100,
    agent_types: tuple[str, ...] = ("zero_shot",),
    seed: int = SEED,
) -> list[AuditRow]:
    bank = _load_bank()
    theta = _load_theta()
    items = _load_items()

    # one pass to find which (cell, task) responses exist and are usable
    usable: dict[tuple[str, str], dict] = {}
    task_ids: set[str] = set()
    with open(RESPONSES_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get("agent_type", "zero_shot") not in agent_types:
                continue
            if r.get("solver_error_type") or not (r.get("raw_response") or "").strip():
                continue
            if r["task_id"] not in bank:
                continue
            usable[(r["examinee_id"], r["task_id"])] = r
            task_ids.add(r["task_id"])

    chosen = set(select_items(bank, task_ids, n_items, seed))

    rows = []
    for (examinee_id, task_id), r in sorted(usable.items()):
        if task_id not in chosen:
            continue
        item = items.get(task_id)
        if item is None:
            continue
        rows.append(
            AuditRow(
                examinee_id=examinee_id,
                task_id=task_id,
                model=r["model"],
                strictness=r["strictness"],
                agent_type=r.get("agent_type", "zero_shot"),
                raw_response=r["raw_response"],
                parsed_result=int(r["parsed_result"]),
                judge_reason=r.get("judge_reason") or "",
                b=bank[task_id],
                theta=theta.get(examinee_id, float("nan")),
                domain=item["domain"],
                context=item["context"],
                question=item["question"],
                gold_standard=item["gold_standard"],
            )
        )
    return rows

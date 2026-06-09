"""
autoevolve.py — Iterative augmented-solver improvement loop.

Each iteration:
  1. Load current solver state and item bank statistics
  2. Build a failure report (domain rates, b-band rates, sample failure reasons)
  3. Ask an LLM to self-critique and propose one targeted change to the solver config
  4. Test the proposal on a stratified sample of items from the bank
  5. If delta_theta >= threshold: adopt the config, git commit + push
  6. Generate new calibration items with the current best solver
  7. Log everything to evolve.log and evolve_history.jsonl

What the LLM is allowed to change each iteration:
  - system_prompt
  - user_template (prompt structure, reference toolkit content)
  - model (any Groq-available model)
  - max_tokens

What never changes automatically:
  - Rasch math, judge prompts, database schema, item generation logic

Usage:
  source venv/bin/activate
  export GROQ_API_KEY=<your_key>
  python3 autoevolve.py --iterations 10 --test-items 15 --min-delta 0.05
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import openai

from calibrator import _load_solver_config, run_champion_solver
from database import Database
from irt_parameters import estimate_theta_mle, THETA_AUGMENTED


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOLVER_CONFIG_PATH  = Path(__file__).parent / "solver_config.json"
EVOLVE_LOG_PATH     = Path(__file__).parent / "evolve.log"
EVOLVE_HISTORY_PATH = Path(__file__).parent / "evolve_history.jsonl"

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "deepseek-r1-distill-llama-70b",
    "qwen-qwq-32b",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

PROPOSER_MODEL = "llama-3.3-70b-versatile"   # model used to generate proposals
JUDGE_MODEL    = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(EVOLVE_LOG_PATH, "a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------

def build_failure_report(db: Database) -> dict:
    """
    Analyse the calibration_results table and return a structured failure
    report for the augmented solver.
    """
    with db._connect() as conn:
        rows = conn.execute("""
            SELECT
                t.domain,
                c.retention_reason,
                c.augmented_pass,
                c.irt_b,
                c.p_augmented
            FROM calibration_results c
            JOIN tasks t ON t.task_id = c.task_id
            ORDER BY c.calibrated_at DESC
        """).fetchall()

    # Per-domain augmented failure rate
    domain_counts: dict = {}
    for r in rows:
        d = r["domain"] or "unknown"
        if d not in domain_counts:
            domain_counts[d] = {"total": 0, "aug_fail": 0}
        domain_counts[d]["total"] += 1
        if not r["augmented_pass"]:
            domain_counts[d]["aug_fail"] += 1

    domain_failure_rates = {
        d: round(v["aug_fail"] / v["total"], 3) if v["total"] else 0
        for d, v in domain_counts.items()
        if not d.startswith("easy_")   # exclude easy domains from signal
    }

    # b-band failure rates (only rows with a b value)
    bands = {"(-inf,0)": [0, 0], "[0,1)": [0, 0], "[1,2)": [0, 0],
             "[2,3)": [0, 0], "[3,5)": [0, 0], "[5,inf)": [0, 0]}

    def band(b):
        if b < 0:   return "(-inf,0)"
        if b < 1:   return "[0,1)"
        if b < 2:   return "[1,2)"
        if b < 3:   return "[2,3)"
        if b < 5:   return "[3,5)"
        return "[5,inf)"

    for r in rows:
        if r["irt_b"] is None:
            continue
        b = band(r["irt_b"])
        bands[b][0] += 1
        if not r["augmented_pass"]:
            bands[b][1] += 1

    b_band_failure_rates = {
        k: round(v[1] / v[0], 3) if v[0] else None
        for k, v in bands.items()
    }

    # Sample soft p_augmented scores for failed items (proxy for failure signal)
    failure_reasons = []
    for r in rows:
        if not r["augmented_pass"] and r["p_augmented"] is not None:
            failure_reasons.append(
                f"domain={r['domain']} b={r['irt_b']} p_aug={round(r['p_augmented'], 3)}"
            )
            if len(failure_reasons) >= 20:
                break

    # Augmented theta from current bank
    resp_vectors = db.fetch_response_vectors()
    aug_responses = resp_vectors.get("augmented", [])
    current_theta = estimate_theta_mle(aug_responses, theta_init=THETA_AUGMENTED)

    # Too-hard item count (structural signal)
    with db._connect() as conn:
        n_too_hard = conn.execute(
            "SELECT COUNT(*) FROM calibration_results WHERE retention_reason = 'discarded_too_hard'"
        ).fetchone()[0]
        n_retained = conn.execute(
            "SELECT COUNT(*) FROM calibration_results WHERE is_retained = 1"
        ).fetchone()[0]
        n_total = conn.execute(
            "SELECT COUNT(*) FROM calibration_results"
        ).fetchone()[0]

    return {
        "n_total_items":         n_total,
        "n_retained":            n_retained,
        "n_too_hard":            n_too_hard,
        "augmented_theta":       current_theta,
        "domain_failure_rates":  domain_failure_rates,
        "b_band_failure_rates":  b_band_failure_rates,
        "sample_failure_reasons": failure_reasons[:10],
    }


# ---------------------------------------------------------------------------
# Proposal generation (LLM self-questions)
# ---------------------------------------------------------------------------

_PROPOSER_SYSTEM = """
You are an AI research engineer tasked with improving a biopharma regulatory compliance solver.
The solver answers questions about FDA regulations (21 CFR Part 11, GCP, GMP, informed consent, promotional review).

The solver runs as a 3-step PIPELINE:
  Step 0 (triage)    — maps regulatory domains, lists suspicious elements, checks for Red Herrings
  Step 1 (analysis)  — section-level CFR/ICH citation and VIOLATION/COMPLIANT/UNCERTAIN ruling per element
  Step 2 (verdict)   — synthesizes the final answer: primary violation, Red Herring explanation, remediation

You will receive:
1. A failure report with population-level stats from the item bank
2. Champion domain pass rates — the ACTUAL pass rates of the current pipeline solver by domain
3. The current pipeline configuration
4. A blacklist of (change_type, step, model) combos that have already been tried and rejected
5. A history of past proposals and their delta_theta outcomes

Your job: identify the most likely root cause of failures and propose ONE concrete, targeted change.

Failure patterns by step:
  - Triage failures: Red Herring not flagged, suspicious elements missed → fix step 0 prompt
  - Analysis failures: wrong CFR section, inverted VIOLATION/COMPLIANT rulings → fix step 1 prompt/template/model
  - Verdict failures: correct analysis but wrong synthesis, section granularity too coarse → fix step 2 prompt

DIVERSITY RULES — you MUST follow these:
1. NEVER propose anything in the BLACKLIST. If a (change_type, step, model) combo is blacklisted, it is forbidden.
2. If step_model for step 1 has been blacklisted for all available models, you MUST switch to
   step_prompt or step_template changes on step 0, 1, or 2.
3. If all step_model changes have been exhausted, escalate to prompt/template improvements:
   - step 0: strengthen the Red Herring detection instructions or add domain-specific triage hints
   - step 1: add explicit regulatory reference lists, stricter citation format, or self-verification step
   - step 2: tighten the output format or add a "double-check against step 1 primary violation" instruction
4. Look at the champion domain pass rates. The LOWEST-RATE domain should guide which step to fix:
   - Informed consent / GCP: often a triage problem (Red Herring detection) → fix step 0
   - 21 CFR Part 11 / GMP: often a citation problem (wrong section level) → fix step 1 or 2
5. Be self-critical: state what assumption might be wrong about your proposal.

Available Groq models (for step_model changes):
  llama-3.3-70b-versatile       (fast general reasoning)
  deepseek-r1-distill-llama-70b (reasoning model, strong multi-step; <think> tags are stripped)
  qwen-qwq-32b                  (reasoning model, strong structured analysis)

Respond in EXACTLY this JSON format (no markdown, no extra text):
{
  "hypothesis": "one sentence describing the root cause of current failures",
  "change_type": "one of: step_prompt | step_template | step_model | step_tokens",
  "step_index": <0, 1, or 2>,
  "description": "one sentence describing the specific change",
  "self_critique": "one sentence on what could go wrong with this proposal",
  "new_system_prompt": "<full updated system prompt for that step, or null if unchanged>",
  "new_user_template": "<full updated user template for that step, or null if unchanged>",
  "new_model": "<model id for that step, or null if unchanged>",
  "new_max_tokens": <integer for that step, or null if unchanged>
}
"""

_PROPOSER_USER = """
--- POPULATION FAILURE REPORT (from item bank, 15-profile calibration) ---
{failure_report}

--- CHAMPION SOLVER PASS RATES (actual pass rates of the current pipeline on last test sample) ---
{champion_rates}

--- CURRENT PIPELINE CONFIG ---
{pipeline_summary}

--- BLACKLIST (forbidden — already tried and rejected, do NOT repeat) ---
{blacklist_str}

--- PROPOSAL HISTORY (most recent first) ---
{history}

Based on the champion pass rates (not the population report), which domain is the weakest?
Which pipeline step is most likely causing that domain's failures?
Propose ONE change — and ensure it is NOT in the blacklist.
"""


def _format_pipeline_summary(config: dict) -> str:
    """Render the current solver config as a readable summary for the proposer."""
    if config.get("mode") == "pipeline":
        parts = []
        for i, step in enumerate(config["pipeline"]):
            parts.append(
                f"Step {i} ({step.get('name', f'step{i}')})\n"
                f"  model={step.get('model', 'default')}  "
                f"max_tokens={step.get('max_tokens', '?')}  "
                f"temperature={step.get('temperature', '?')}\n"
                f"  system_prompt: {step['system_prompt'][:300]}\n"
                f"  user_template: {step['user_template'][:300]}"
            )
        return "\n\n".join(parts)
    # Legacy single-call format
    return (
        f"mode=single  model={config.get('model', '?')}  "
        f"max_tokens={config.get('max_tokens', '?')}\n"
        f"system_prompt: {config.get('system_prompt', '')[:300]}\n"
        f"user_template: {config.get('user_template', '')[:300]}"
    )


def generate_proposal(
    client:               openai.OpenAI,
    failure_report:       dict,
    current_config:       dict,
    history:              list[dict],
    champion_domain_rates: dict | None = None,
    blacklist:            set  | None = None,
) -> dict | None:
    """Ask the LLM to self-question and propose one improvement."""
    history_str = "\n".join(
        f"Iter {h['iteration']}: [step={h.get('step_index', '?')} {h['change_type']}] "
        f"{h['description']} "
        f"→ delta_theta={h.get('delta_theta', 'N/A')} ({'ADOPTED' if h.get('adopted') else 'REJECTED'})"
        for h in history[-10:]
    ) or "No history yet."

    if champion_domain_rates:
        rates_str = "  " + "\n  ".join(
            f"{d}: {round(r*100)}% pass" if r is not None else f"{d}: no data"
            for d, r in sorted(champion_domain_rates.items(), key=lambda x: (x[1] or 1))
        )
    else:
        rates_str = "  Not yet available (first iteration)."

    if blacklist:
        bl_lines = [
            f"  step_model / step={si} / model={mdl}"
            if ct == "step_model" else f"  {ct} / step={si}"
            for ct, si, mdl in sorted(blacklist)
        ]
        blacklist_str = "\n".join(bl_lines) or "  (none yet)"
    else:
        blacklist_str = "  (none yet)"

    prompt = _PROPOSER_USER.format(
        failure_report=json.dumps(
            {k: v for k, v in failure_report.items()
             if k in ("n_total_items", "n_retained", "n_too_hard", "domain_failure_rates", "b_band_failure_rates")},
            indent=2,
        ),
        champion_rates=rates_str,
        pipeline_summary=_format_pipeline_summary(current_config),
        blacklist_str=blacklist_str,
        history=history_str,
    )

    try:
        msg = client.chat.completions.create(
            model=PROPOSER_MODEL,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": _PROPOSER_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = msg.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = raw[:-3]
        return json.loads(raw)
    except Exception as exc:
        log(f"  [Proposer] ERROR generating proposal: {exc}")
        return None


# ---------------------------------------------------------------------------
# Proposal testing
# ---------------------------------------------------------------------------

def build_test_config(proposal: dict, current_config: dict) -> dict:
    """
    Apply the proposed changes on top of the current config.

    For pipeline configs, changes are applied to the specific step identified
    by proposal["step_index"].  For legacy single-call configs, changes are
    applied at the top level.
    """
    import copy

    cfg = copy.deepcopy(current_config)

    def _str_or_none(v):
        return v if isinstance(v, str) and v.strip().lower() not in ("null", "none", "") else None

    def _int_or_none(v):
        try:
            return int(v) if v is not None and str(v).strip().lower() not in ("null", "none") else None
        except (ValueError, TypeError):
            return None

    new_sys  = _str_or_none(proposal.get("new_system_prompt"))
    new_tmpl = _str_or_none(proposal.get("new_user_template"))
    new_mdl  = _str_or_none(proposal.get("new_model"))
    new_tok  = _int_or_none(proposal.get("new_max_tokens"))

    if cfg.get("mode") == "pipeline":
        step_index = proposal.get("step_index")
        try:
            step_index = int(step_index)
        except (TypeError, ValueError):
            step_index = None

        if step_index is None or not (0 <= step_index < len(cfg["pipeline"])):
            log(f"  [Config] Invalid step_index={step_index!r} — skipping pipeline change.")
            return cfg

        step = cfg["pipeline"][step_index]
        if new_sys:  step["system_prompt"] = new_sys
        if new_tmpl: step["user_template"]  = new_tmpl
        if new_tok:  step["max_tokens"]     = new_tok
        if new_mdl:
            if new_mdl in GROQ_MODELS:
                step["model"] = new_mdl
            else:
                log(f"  [Config] Proposed model '{new_mdl}' not in GROQ_MODELS — ignoring.")
    else:
        # Legacy single-call path
        if new_sys:  cfg["system_prompt"] = new_sys
        if new_tmpl: cfg["user_template"]  = new_tmpl
        if new_tok:  cfg["max_tokens"]     = new_tok
        if new_mdl:
            if new_mdl in GROQ_MODELS:
                cfg["model"] = new_mdl
            else:
                log(f"  [Config] Proposed model '{new_mdl}' not in GROQ_MODELS — ignoring.")

    return cfg


def sample_test_items(db: Database, n: int, seed: int = 42) -> list[dict]:
    """
    Sample N retained items from the bank, spread uniformly across four b-bands
    to maximise Fisher information in the theta MLE.

    Only retained items are used — discarded_too_hard items (b ≈ 4.6, P=0)
    provide near-zero Fisher information because no solver configuration can
    pass them, so they contribute only noise to the delta-theta estimate.

    Bands (equal quota, remainder fills from the most populated):
      Band A: b in [-5, 0)    — easy/negative difficulty
      Band B: b in [0, 1)     — moderate
      Band C: b in [1, 2)     — hard
      Band D: b in [2, 3)     — very hard but solvable
    Items with b ≥ 3 are excluded (too far above champion theta to be informative).
    """
    all_tasks = db.fetch_all_tasks()
    task_map  = {t["task_id"]: t for t in all_tasks}

    with db._connect() as conn:
        rows = conn.execute("""
            SELECT task_id, irt_b, retention_reason
            FROM calibration_results
            WHERE is_retained = 1
              AND irt_b IS NOT NULL
              AND irt_b < 3.0
            ORDER BY irt_b
        """).fetchall()

    bands: dict[str, list] = {"A": [], "B": [], "C": [], "D": []}
    for r in rows:
        b = r["irt_b"]
        if b < 0:
            bands["A"].append(r)
        elif b < 1:
            bands["B"].append(r)
        elif b < 2:
            bands["C"].append(r)
        else:
            bands["D"].append(r)

    rng    = random.Random(seed)
    quota  = n // 4
    result = []
    leftover = n

    for band_rows in bands.values():
        take = min(len(band_rows), quota)
        result.extend(rng.sample(band_rows, take))
        leftover -= take

    # Fill remainder from the largest band(s)
    if leftover > 0:
        remaining = [r for r in rows if r not in result]
        rng.shuffle(remaining)
        result.extend(remaining[:leftover])

    items = []
    for r in result:
        task = task_map.get(r["task_id"])
        if task:
            task = dict(task)
            task["is_easy"] = r["irt_b"] < 0
            items.append(task)
    return items


def test_proposal(
    client:        openai.OpenAI,
    proposed_cfg:  dict,
    test_items:    list[dict],
    current_theta: float,
    db:            Database,
    verbose:       bool = True,
) -> tuple[float, float, dict]:
    """
    A/B test: run both the current and proposed champion configs on the same
    items and compare their θ estimates.

    Also tracks per-domain pass rates for the CURRENT config at zero extra
    cost (reuses the current-config run that the A/B test already performs).

    Returns (proposed_theta, delta_theta, champion_domain_rates).
    champion_domain_rates: {domain: pass_rate} for the current config on this sample.
    """
    current_cfg = _load_solver_config()

    current_responses:  list[tuple[float, bool]] = []
    proposed_responses: list[tuple[float, bool]] = []

    # Per-domain tracking for the current config
    domain_counts: dict[str, list[int]] = {}   # {domain: [n_pass, n_total]}

    for task in test_items:
        with db._connect() as conn:
            row = conn.execute(
                "SELECT irt_b FROM calibration_results WHERE task_id = ? AND irt_b IS NOT NULL",
                (task["task_id"],),
            ).fetchone()
        if not row:
            continue
        b = row["irt_b"]

        domain = task.get("domain", "unknown")
        if domain not in domain_counts:
            domain_counts[domain] = [0, 0]

        try:
            cur_passed, _ = run_champion_solver(
                client, task, current_cfg, JUDGE_MODEL, verbose=False
            )
            current_responses.append((b, cur_passed))
            domain_counts[domain][0] += int(cur_passed)
            domain_counts[domain][1] += 1
        except Exception as exc:
            log(f"    [Test/Current]  Error on {task['task_id']}: {exc}")
            continue

        try:
            prop_passed, _ = run_champion_solver(
                client, task, proposed_cfg, JUDGE_MODEL, verbose=verbose
            )
            proposed_responses.append((b, prop_passed))
        except Exception as exc:
            log(f"    [Test/Proposed] Error on {task['task_id']}: {exc}")

    # Champion domain pass rates (exclude easy domains — not useful signal)
    champion_domain_rates = {
        d: round(v[0] / v[1], 3) if v[1] > 0 else None
        for d, v in domain_counts.items()
        if not d.startswith("easy_")
    }
    log(f"  [Champion] Domain pass rates: {champion_domain_rates}")

    if not current_responses or not proposed_responses:
        return current_theta, 0.0, champion_domain_rates

    current_test_theta  = estimate_theta_mle(current_responses,  theta_init=current_theta)
    proposed_test_theta = estimate_theta_mle(proposed_responses, theta_init=current_theta)
    delta_theta = round(proposed_test_theta - current_test_theta, 4)

    log(f"  [Test] θ_current={current_test_theta:.4f}  θ_proposed={proposed_test_theta:.4f}")

    return proposed_test_theta, delta_theta, champion_domain_rates


# ---------------------------------------------------------------------------
# Config persistence and git
# ---------------------------------------------------------------------------

def save_config(cfg: dict, iteration: int, theta: float, description: str) -> None:
    cfg_out = dict(cfg)
    cfg_out["version"]           = iteration
    cfg_out["iteration"]         = iteration
    cfg_out["theta_at_adoption"] = theta
    cfg_out["adopted_at"]        = datetime.now(timezone.utc).isoformat()
    cfg_out["description"]       = description
    with open(SOLVER_CONFIG_PATH, "w") as f:
        json.dump(cfg_out, f, indent=2, ensure_ascii=False)


def append_history(entry: dict) -> None:
    with open(EVOLVE_HISTORY_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_history() -> list[dict]:
    if not EVOLVE_HISTORY_PATH.exists():
        return []
    history = []
    with open(EVOLVE_HISTORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return history


def git_commit_push(iteration: int, delta_theta: float, description: str) -> bool:
    """Stage changed files, commit, and push to origin/dev. Returns True on success."""
    repo_root = str(Path(__file__).parent)
    files_to_stage = ["solver_config.json", "evolve_history.jsonl", "evolve.log"]

    try:
        # Ensure we are on dev
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root, text=True
        ).strip()
        if branch != "dev":
            log(f"  [Git] WARNING: on branch '{branch}', expected 'dev'. Skipping push.")
            return False

        # Stage files
        for f in files_to_stage:
            path = os.path.join(repo_root, f)
            if os.path.exists(path):
                subprocess.run(["git", "add", path], cwd=repo_root, check=True)

        # Commit
        msg = (
            f"autoevolve iter {iteration:03d}: Δθ=+{delta_theta:.3f} — {description}\n\n"
            f"Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
        )
        result = subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=repo_root, capture_output=True, text=True
        )
        if result.returncode != 0:
            if "nothing to commit" in result.stdout + result.stderr:
                log("  [Git] Nothing to commit, skipping push.")
                return True
            log(f"  [Git] Commit failed: {result.stderr.strip()}")
            return False

        # Push
        push = subprocess.run(
            ["git", "push", "origin", "dev"],
            cwd=repo_root, capture_output=True, text=True
        )
        if push.returncode != 0:
            log(f"  [Git] Push failed: {push.stderr.strip()}")
            return False

        log(f"  [Git] Committed and pushed iter {iteration:03d}.")
        return True

    except Exception as exc:
        log(f"  [Git] Exception during commit/push: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def evolve(
    n_iterations:  int   = 10,
    n_test_items:  int   = 40,
    min_delta:     float = 0.50,
    generate_new:  bool  = True,
    n_generate:    int   = 5,
    db_path:       str   = "compliance_bank.db",
    verbose:       bool  = False,
) -> None:

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY not set. Run: export GROQ_API_KEY=<your_key>")

    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    db = Database(db_path)

    log("=" * 70)
    log(f"autoevolve started: {n_iterations} iterations, test_items={n_test_items}, min_delta={min_delta}")
    log("=" * 70)

    history = load_history()

    # Per-run state: blacklist of rejected (change_type, step_index, new_model) tuples
    # and the champion domain rates from the most recent A/B test.
    blacklist:             set  = set()
    champion_domain_rates: dict = {}

    for iteration in range(1, n_iterations + 1):
        log(f"\n{'─'*60}")
        log(f"ITERATION {iteration}/{n_iterations}")
        log(f"{'─'*60}")

        try:
            # ── 1. Current state ─────────────────────────────────────────
            current_config = _load_solver_config()
            failure_report = build_failure_report(db)
            current_theta  = failure_report["augmented_theta"]

            log(f"  Current θ_augmented = {current_theta:.4f}")
            log(f"  Population domain failure rates: {failure_report['domain_failure_rates']}")
            log(f"  Champion domain pass rates (last test): {champion_domain_rates or 'not yet measured'}")
            log(f"  Blacklist size: {len(blacklist)}")

            # ── 2. Generate proposal ──────────────────────────────────────
            log("  [Proposer] Generating improvement proposal…")
            proposal = generate_proposal(
                client, failure_report, current_config, history,
                champion_domain_rates=champion_domain_rates,
                blacklist=blacklist,
            )

            if proposal is None:
                log("  [Proposer] No valid proposal generated. Skipping iteration.")
                continue

            log(f"  [Proposer] Hypothesis  : {proposal.get('hypothesis', '')}")
            log(f"  [Proposer] Change type : {proposal.get('change_type', '')}  step={proposal.get('step_index', '?')}")
            log(f"  [Proposer] Description : {proposal.get('description', '')}")
            log(f"  [Proposer] Self-critique: {proposal.get('self_critique', '')}")

            # ── 3. Build and test proposed config ─────────────────────────
            proposed_config = build_test_config(proposal, current_config)
            test_items      = sample_test_items(db, n=n_test_items, seed=iteration)

            if not test_items:
                log("  [Test] No test items available. Run main.py first to build the bank.")
                continue

            log(f"  [Test] Testing on {len(test_items)} items…")
            new_theta, delta_theta, champion_domain_rates = test_proposal(
                client, proposed_config, test_items, current_theta, db, verbose=verbose
            )
            log(f"  [Test] Δθ = {delta_theta:+.4f}  (threshold: {min_delta:+.4f})")

            adopted = delta_theta >= min_delta

            # ── 4. Adopt or reject ────────────────────────────────────────
            if adopted:
                log(f"  [Decision] ADOPTED (Δθ={delta_theta:+.4f} ≥ {min_delta})")
                save_config(
                    proposed_config, iteration, new_theta,
                    proposal.get("description", "")
                )
                git_commit_push(iteration, delta_theta, proposal.get("description", ""))
            else:
                log(f"  [Decision] REJECTED (Δθ={delta_theta:+.4f} < {min_delta})")
                # Add to blacklist so the proposer doesn't repeat it
                bl_key = (
                    proposal.get("change_type", ""),
                    proposal.get("step_index"),
                    proposal.get("new_model"),
                )
                blacklist.add(bl_key)

            # ── 5. Log to history ─────────────────────────────────────────
            entry = {
                "iteration":    iteration,
                "timestamp":    datetime.now(timezone.utc).isoformat(),
                "hypothesis":   proposal.get("hypothesis", ""),
                "change_type":  proposal.get("change_type", ""),
                "step_index":   proposal.get("step_index"),
                "description":  proposal.get("description", ""),
                "self_critique": proposal.get("self_critique", ""),
                "theta_before": current_theta,
                "theta_after":  new_theta,
                "delta_theta":  delta_theta,
                "adopted":      adopted,
                "n_test_items": len(test_items),
            }
            history.append(entry)
            append_history(entry)

            # ── 6. Generate new bank items with current best solver ────────
            if generate_new:
                log(f"  [Generate] Running main.py to add {n_generate} new items…")
                try:
                    subprocess.run(
                        [sys.executable, "main.py", f"--n={n_generate}", "--easy-n=1", "--quiet"],
                        cwd=str(Path(__file__).parent),
                        timeout=300,
                    )
                except subprocess.TimeoutExpired:
                    log("  [Generate] main.py timed out after 5 min.")
                except Exception as exc:
                    log(f"  [Generate] Error: {exc}")

        except KeyboardInterrupt:
            log("\nInterrupted by user.")
            break
        except Exception as exc:
            log(f"  [ERROR] Iteration {iteration} failed: {exc}")
            log(traceback.format_exc())
            continue

    log("\n" + "=" * 70)
    log("autoevolve complete.")
    adopted_count = sum(1 for h in history if h.get("adopted"))
    if history:
        final_theta = history[-1].get("theta_after", "?")
        first_theta = history[0].get("theta_before", "?")
        log(f"  Adopted {adopted_count}/{len(history)} proposals")
        log(f"  θ trajectory: {first_theta} → {final_theta}")
    log("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative augmented-solver evolution loop")
    parser.add_argument("--iterations",   type=int,   default=10,   help="Number of evolution iterations")
    parser.add_argument("--test-items",   type=int,   default=40,   help="Items sampled per test (default 40; ~±0.5 SE with band sampling)")
    parser.add_argument("--min-delta",    type=float, default=0.50, help="Min delta_theta to adopt a proposal (default 0.50 ≈ 1×SE with 40 band-sampled items)")
    parser.add_argument("--no-generate",  action="store_true",      help="Skip generating new bank items each iteration")
    parser.add_argument("--n-generate",   type=int,   default=5,    help="New hard items to generate per iteration")
    parser.add_argument("--db",           default="compliance_bank.db")
    parser.add_argument("--verbose",      action="store_true",      help="Print calibration detail per item")
    args = parser.parse_args()

    evolve(
        n_iterations = args.iterations,
        n_test_items = args.test_items,
        min_delta    = args.min_delta,
        generate_new = not args.no_generate,
        n_generate   = args.n_generate,
        db_path      = args.db,
        verbose      = args.verbose,
    )

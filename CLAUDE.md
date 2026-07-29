# CLAUDE.md — BIO_Compliance_IRT

Persistent context for Claude Code sessions. Update this file when significant decisions change.

---
**CORE DIRECTIVE: SELF-DOCUMENTATION**
Whenever you are asked to fix a bug, change a mathematical formula, or alter the architecture of this project, you MUST independently update this `CLAUDE.md` file to reflect the new state of the project before you commit the changes to git. Do not wait for the user to explicitly ask you to update the documentation.
## 1. Project Overview

This project builds an **automated bio-compliance Item Response Theory (IRT) question bank** for evaluating AI systems (and potentially human experts) on biopharma regulatory knowledge.

The core loop:
1. An LLM **generates** compliance scenarios across 5 regulatory domains (FDA 21 CFR Part 11, GCP deviations, promotional review, GMP, informed consent)
2. **15 synthetic AI solver profiles** attempt each scenario — spanning a spectrum from "general layperson" (temp 0.1–0.8) to "senior FDA auditor" (temp 0.1–0.8) across 5 expertise tiers
3. An LLM **judge** grades each response PASS or FAIL
4. Item difficulty `b` is estimated directly from the empirical **pass rate P** using `b = ln((1−P)/P)` (logit formula, P clipped to [0.01, 0.99])
5. Items are classified by pass-rate thresholds: P < 0.05 → too hard, P > 0.95 → easy, else retained (widened 2026-04-14)
6. Results persist to **SQLite** for cumulative calibration across runs
7. Once enough items accumulate (configurable threshold), solver **ability (θ) is estimated via MLE**

A parallel **real-world track** ingests actual FDA warning letters and converts them into calibration items, tagged `source_type="real"`. This grounds the synthetic bank in actual regulatory enforcement findings.

**Current bank size (Phase 2 frozen bank, 2026-05-10):**
- Synthetic frozen: 1,191 items
- Real-world frozen: 93 items (from FDA warning letters)
- Total frozen (healthy): 1,284 items in `evaluator/output/phase2_frozen_bank.jsonl`
- Input to Phase 2: 1,823 items (539 dropped by QC)

---

## 2. Architecture

There are now **two parallel pipelines** in this repo:

- **Legacy bank pipeline** (`main.py`, `task_generator.py`, `calibrator.py`, ...) — generates synthetic compliance items, runs 15 solver profiles, computes b from pass rate, persists to `compliance_bank.db`. Still the source of new items.
- **Phase 1 calibration pipeline** (`run_phase1.py` + `agents/` + `arena/` + `evaluator/` + `harness/` + `configs/`) — reads items from `compliance_bank.db` (read-only), runs a 45-cell virtual-examinee grid, fits Rasch (1PL) via `girth`, anchors a baseline examinee at θ=0. Produces per-item b estimates in `evaluator/output/`.

### Repo layout

```
BIO_Compliance_IRT/               ← repo root
  # --- Legacy bank pipeline (still the item generator) -----------------
  main.py                         ← CLI entry point for legacy pipeline
  task_generator.py               ← Stage 1: LLM generates compliance tasks
  calibrator.py                   ← Stage 2: 15 solver profiles + judge → pass rate → logit b
  legacy_evaluator.py             ← LLM-as-judge (strict and lenient modes); renamed from evaluator.py to avoid clash with evaluator/ package
  autoevolve.py                   ← Iterative solver evolution loop (IRT fitness via frozen bank)
  irt_parameters.py               ← Rasch math, outcome classification, theta MLE
  database.py                     ← SQLite persistence
  fda_importer.py                 ← Real-world FDA warning letter ingestion
  build_bank.sh / build_real.sh   ← background builders
  compliance_bank.db              ← shared SQLite bank (read by both pipelines)

  # --- Phase 1 / 2 / 3 calibration pipeline ----------------------------
  run_phase1.py                   ← Phase 1 entry point (yaml-driven Arena → Rasch fit + anchor)
  run_phase2.py                   ← Phase 2 entry point (Rasch refit + classical QC → frozen bank)
  run_phase3.py                   ← Phase 3a entry point (offline Δθ on frozen bank, dual-judge)
  run_phase3_rejudge_lenient.py   ← Phase 3a helper: re-grade Phase 1 responses with lenient judge
  run_phase3b.py                  ← Phase 3b entry point (agent-variant grid → Rasch θ + Δθ vs zero_shot)
  configs/phase1_baseline.yaml    ← 45-cell grid + baseline examinee config
  configs/phase3b_variants.yaml   ← Phase 3b 16-cell grid (2 models × t=0.4 × 2 strictness × 4 agent_types)
  agents/                         ← ZeroShotAgent + RetrievalAgent + CriticAgent + StepDecompositionAgent
  arena/                          ← grid generator, item loader, resumable runner, jsonl schema
  evaluator/                      ← Rasch (1PL) MML fit + 2PL QC + response matrix builder
  harness/                        ← GroqClient (reasoning-model handling) + Judge + CFRStore + errors
  scripts/build_cfr_index.py      ← one-time eCFR download + all-MiniLM-L6-v2 embedding index
  data/cfr_index/                 ← gitignored CFR embedding index (embeddings.npy + sections.jsonl)
  logs/arena_runs/<run_id>/responses.jsonl  ← raw audit log (one row per examinee×item)
  evaluator/output/<run_id>_phase1_b.jsonl  ← anchored b estimates
  evaluator/output/phase2_frozen_bank.jsonl ← Phase 2 healthy items (b + pb + infit + outfit)
  evaluator/output/phase3a_strictness_deltatheta.json  ← Phase 3a Δθ tables (strict + lenient)
  evaluator/output/phase3b_agent_deltatheta.json       ← Phase 3b Δθ per agent type (after run)
  evaluator/pyirt_fit.py          ← py-irt 1PL fit with posterior SD (b_se) — uncertainty path

  # --- Judge calibration audit (IRT judge vs DeepEval/RAGAS) ----------
  judge_audit/                    ← standalone audit module (no restructure of main pipeline)
    __init__.py
    backends.py                   ← GroqDeepEvalLLM (DeepEvalBaseLLM subclass) + make_ragas_llm
    sample.py                     ← AuditRow dataclass + build_audit_set (Phase 3b responses join)
    metrics.py                    ← run_metrics(): 4 metrics, ThreadPoolExecutor, resumable JSONL
    analysis.py                   ← 4 analyses → judge_audit_report.json + printed table
  run_judge_audit.py              ← CLI: --smoke / --n-items / --parallelism / --analyze-only
  evaluator/output/judge_audit_scores.jsonl       ← (metric, examinee_id, task_id) scored rows
  evaluator/output/judge_audit_report.json        ← AUC, Spearman, slopes, disagreement counts
  evaluator/output/judge_audit_disagreements.jsonl ← top-20 high-score-but-FAIL + low-score-but-PASS

  # --- arXiv paper: lives on the `paper` branch, NOT on main -----------
  paper/main.tex                  ← document root (article class)
  paper/sections/00..09_*.tex     ← per-section .tex (Part 1 instrument, Part 2 Δθ)
  paper/sections/07b_phase3b.tex  ← Phase 3b agent-variant section
  paper/sections/07c_judge_audit.tex   ← judge audit vs G-Eval/RAGAS (added 2026-07-29)
  paper/sections/07d_expert_review.tex ← expert-review protocol, pre-committed (2026-07-29)
  paper/references.bib            ← 19 entries, ALL verified against primary sources 2026-07-29
  paper/figures/gen_fig1..4_*.py  ← matplotlib scripts → fig{1..4}.pdf/png (no API calls)
  paper/figures/_common.py        ← shared loaders + analytic SE = 1/√(N·P(1−P))
  scripts/real_vs_synthetic_test.py ← stdlib Mann-Whitney U on frozen bank (no deps, no API)

  venv/                           ← Python virtualenv (gitignored)
  .env                            ← GROQ_API_KEY (gitignored)
  CLAUDE.md                       ← this file
  METHODOLOGY.md                  ← methodological decision record (decision / alternatives / rationale / limitations per choice)
```

### Note on repo history
The repo previously had a nested `autoresearch_IRT/` subdirectory inside a TypeScript monorepo scaffold (pnpm, Express, Drizzle). Both were consolidated into this flat structure. The TypeScript scaffold (lib/, artifacts/, scripts/) was removed entirely — it had no domain logic.

### Python module dependencies

Legacy bank pipeline:
```
main.py
  ├── task_generator.py   (generates hard + easy tasks)
  ├── calibrator.py
  │     ├── legacy_evaluator.py  (strict or lenient judge; renamed from evaluator.py)
  │     └── irt_parameters.py
  ├── database.py
  └── fda_importer.py     (print_comparison)

fda_importer.py           (standalone CLI + helper used by main.py)
  ├── calibrator.py
  └── database.py
```

Phase 1 calibration pipeline:
```
run_phase1.py
  ├── arena/loader.py      (read-only DB → Item list, optional stratified sample)
  ├── arena/grid.py        (Cartesian product → ExamineeConfig list)
  ├── arena/runner.py      (resumable parallel loop, writes responses.jsonl)
  │     ├── agents/zero_shot.py   (single-call ZeroShotAgent, 3 strictness levels)
  │     │     └── harness/groq_client.py   (GroqClient + reasoning-model handling)
  │     └── harness/judge.py      (Judge: strict for hard, lenient for easy)
  └── evaluator/
        ├── response_matrix.py    (jsonl → ResponseMatrix)
        ├── rasch.py              (1PL MML fit + anchor_baseline_to_zero)
        ├── rasch_qc.py           (Phase 2 QC: point-biserial + infit/outfit)
        └── twopl.py + qc.py      (DORMANT 2PL path, kept for future re-evaluation)
```

Phase 3a analysis (offline — no new API calls for the strict pass; lenient pass re-grades existing responses):
```
run_phase3.py                          (offline analysis — reads Phase 1 responses + frozen bank)
  ├── evaluator/output/phase2_frozen_bank.jsonl   (1284 anchored b values)
  ├── logs/arena_runs/phase1_baseline/responses.jsonl   (strict-judge verdicts)
  ├── [optional] logs/arena_runs/phase3a_lenient/responses.jsonl   (lenient-judge verdicts)
  └── girth.ability_mle  (per-examinee θ on frozen subset, anchored so baseline θ=0)

run_phase3_rejudge_lenient.py          (only API-calling helper — re-grades existing raw_response with lenient judge)
  ├── arena/loader.py          (compliance_bank.db → question + gold_standard lookup)
  ├── harness/judge.py         (Judge with strict=False)
  └── arena/schema.py          (writes ArenaLogEntry-compatible jsonl for run_phase3.py)
```

Phase 3b agent-variant pipeline (live arena run on the frozen bank):
```
run_phase3b.py
  ├── configs/phase3b_variants.yaml
  ├── arena/grid.py            (agent_type is now a 4th grid axis; zero_shot IDs unchanged)
  ├── arena/runner.py          (optional agent_factory dispatches on examinee.agent_type)
  ├── agents/retrieval.py      (RetrievalAgent: top-k CFR sections → system prompt)
  │     └── harness/cfr_store.py   (CFRStore: numpy cosine sim, index from scripts/build_cfr_index.py)
  ├── agents/critic.py         (CriticAgent: answer → critique → synthesis, 3 calls/item)
  ├── agents/step_decomp.py    (StepDecompositionAgent: 4 sequential scaffolded calls)
  └── girth.ability_mle        (θ per cell on frozen bank, anchored to Phase 1 baseline)
```

---

## 3. Tech Stack

- **Python 3.12** (Homebrew on macOS — externally managed, requires venv)
- **openai SDK** — pointed at Groq's OpenAI-compatible endpoint, not OpenAI directly
- **Groq API** (`https://api.groq.com/openai/v1`) — model: `llama-3.3-70b-versatile`
- **SQLite** (stdlib `sqlite3`) — no ORM, raw SQL, single persistent connection per `Database` instance
- **sentence-transformers** (`all-MiniLM-L6-v2`, ~22 MB, local) — CFR embedding for RetrievalAgent; no API calls
- **venv** at `./venv/` — activate before running anything

### Running the pipelines
```bash
cd /path/to/BIO_Compliance_IRT
source venv/bin/activate
export $(cat .env | xargs)

# Legacy bank pipeline (generates new items)
python3 main.py --n 5 --easy-n 2 --quiet
python3 fda_importer.py --url <url> --max-items 4

# Phase 1 calibration pipeline (calibrates the existing bank)
python3 run_phase1.py                  # full run from configs/phase1_baseline.yaml
python3 run_phase1.py --smoke          # 3 items × 45 examinees sanity check
python3 run_phase1.py --calibrate-only # skip arena, refit IRT on existing log

# Phase 3b agent-variant pipeline (needs the CFR index first)
python3 scripts/build_cfr_index.py --smoke  # Part 11 only + retrieval test queries
python3 scripts/build_cfr_index.py          # full index (21 CFR 11/50/58/211 + 45 CFR 46)
python3 run_phase3b.py --smoke              # 3 items × 16 cells, all 4 agent types
python3 run_phase3b.py                      # full run (~41k solver + ~20k judge calls)
python3 run_phase3b.py --calibrate-only     # skip arena, refit θ on existing log
```

### Background build scripts
```bash
# Synthetic (15 rounds of 5 hard + 2 easy tasks)
nohup bash build_bank.sh &

# Real FDA (Claude finds + imports warning letters autonomously)
nohup bash build_real.sh &

# Monitor
tail -f bank_build.log
tail -f bank_build_real.log
```

---

## 4. Key Conventions

### Task dict schema
```python
{
  "task_id":      "TASK-{DOMAIN_KEY}-{8HEX}",         # synthetic hard
                  "TASK-REAL-{DOMAIN[:8]}-{8HEX}",    # real-world
  "domain":       str,         # e.g. "21cfr11", "gcp_deviation", "easy_training"
  "context":      str,
  "question":     str,
  "gold_standard": str,
  "source_type":  "synthetic" | "real",
  "source_ref":   str | None,  # URL for real items
  "is_easy":      bool,        # runtime only, not persisted to DB
}
```

### Domain keys
- **Hard synthetic**: `21cfr11`, `gcp_deviation`, `promo_review`, `gmp_deviation`, `informed_consent`
- **Easy synthetic**: `easy_training`, `easy_consent`, `easy_batch_record`, `easy_irb`, `easy_backup`
- **Real-world**: domain assigned by LLM during extraction, same keys as hard synthetic

### Outcome classification (15-profile pass rate)
Thresholds widened 2026-04-14 (High-Subtlety bank) to capture near-extreme items.

| Pass rate P | retention_reason | b formula | is_retained |
|---|---|---|---|
| 0.05 ≤ P ≤ 0.95 | `retained` | `ln((1−P)/P)` | True |
| P > 0.95 | `retained_easy` | `ln((1−P)/P)` (negative) | True |
| P < 0.05 | `discarded_too_hard` | `ln((1−P)/P)` (large positive) | False |

*Previous thresholds (pre-2026-04-14): P < 0.10 → too_hard, P > 0.90 → easy.*

`b = ln((1−P_clipped)/P_clipped)` with `P_clipped = clip(P, 0.01, 0.99)`.
`is_retained = True` for both `retained` and `retained_easy` only.
Too-hard items (P < 0.10) still get a b value and enter the θ MLE response vector.

**DB backward compat**: `vanilla_pass`/`vanilla_response`/`p_vanilla` = profile index 0 (layperson_t0.1);
`augmented_pass`/`augmented_response`/`p_augmented` = profile index 14 (fda_auditor_t0.7).

---

## 5. Important Decisions

### Rasch (1PL) with 15-profile population
`a = RASCH_A = 1.0` fixed for all items. `b` is estimated from the empirical pass rate across the 15 synthetic profiles using the population-logit formula:

    b = ln((1 − P) / P)    where P = pass_count / 15, clipped to [0.01, 0.99]

This assumes the 15 profiles are centered at θ = 0 (THETA_POPULATION_MEAN). The formula is derived from the Rasch model: at θ = 0, P(correct) = 1/(1 + exp(b)), so b = −logit(P).

Range: P=0.01 → b ≈ +4.60 (extremely hard); P=0.99 → b ≈ −4.60 (trivially easy).

**Legacy**: Some early DB rows have `a != 1.0` from a 2PL design phase. Identifiable by `a != 1.0`.
**Legacy**: DB rows from the 2-profile era have `vanilla_pass` and `augmented_pass` from the original Vanilla/Augmented solvers. In the 15-profile architecture, these columns are populated from profile index 0 (layperson) and index 14 (expert) respectively for backward compatibility.

### Two judge modes
- **Strict** (`strict=True`): requires section-level citation (e.g., "21 CFR 11.10(e)"). Used for hard synthetic and all real-world items.
- **Lenient** (`strict=False`): requires only correct conclusion + reasoning. Used for easy synthetic items so Vanilla can pass them.

Without the lenient judge, easy items would still get Vanilla FAIL (it never cites sections), defeating the purpose.

### High-Subtlety generation model (2026-04-14)
All hard synthetic items now require:
1. **Red Herring**: at least one compliant process that looks suspicious (e.g., a legacy system correctly running read-only, a delayed but permissible timestamp, a deviation log correctly handled). The Red Herring should misdirect solvers toward flagging a non-violation.
2. **Subtle violation**: the actual compliance gap must be a procedural lapse, signature timing issue, or system-to-system data integrity gap — NOT a blatant, obvious non-compliance.
3. Gold standard must briefly explain why the Red Herring is not the violation.

This is encoded in `_USER_TEMPLATE` (task_generator.py). Easy items use `_USER_TEMPLATE_EASY` which is unchanged (no Red Herring, straightforward).

### Targeted 21 CFR Part 11 generation
`DOMAINS_21CFR11_TARGETED` in `task_generator.py` contains 10 sub-domain prompts focusing on:
- Hybrid Systems (predicate record designation, wet-ink scan signatures, open-system encryption)
- Legacy System Validation (post-1997 modifications, OS patch change control, role access gaps)
- Audit trail completeness (system-initiated changes, migration field-level traceability)
- Electronic signature components (missing meaning field per 21 CFR 11.50(a))
- Predicate rule interplay (21 CFR 58 GLP archive accessibility)

CLI: `python3 main.py --gen-21cfr11 50`

### Full-bank recalibration
`python3 main.py --recalibrate-all` re-evaluates every task in the DB with the current
15-profile population and new 0.05/0.95 thresholds. Deletes old calibration_results rows
for each task and inserts fresh ones. ~30 API calls per task — expect several hours for
large banks. Progress checkpoints every 50 items.

### `is_easy` not persisted
Added in `main.py` before calibration, not stored in DB. Easy-ness is inferred from `retention_reason = 'retained_easy'` or domain key prefix `easy_`.

### Real-world track rationale
Synthetic items are generated by the same LLM that solves them — a closed loop. Real FDA warning letters represent actual enforcement actions.

**Corrected 2026-07-29 (frozen bank, n=93 real / 1,191 synthetic).** Earlier notes claimed "zero real items below b=0" at N=54; that is **false** on the frozen bank — 5 of 93 real items (5.4%) have b < 0, min = −2.00. The honest framing is a *lower-tail shift*, not a separation: 5.4% of real items fall below b=0 versus 31.2% of synthetic, and the easiest real item (−2.00) is well above the easiest synthetic (−4.25).

Mann-Whitney U (tie-corrected, run via `scripts/real_vs_synthetic_test.py`, no API calls):
- mean b: real **+1.98** vs synthetic **+1.04** → gap **+0.94**
- U = 66,516, z = −3.24, **two-sided p = 0.0012**
- Effect size: Cliff's δ = **+0.20**, P(real > synthetic) = 0.60 — statistically clear, modest in magnitude
- Prior "+1.14 at N=54" and the Phase 1 pre-QC "+0.49" figures refer to different item sets/anchorings; do not mix them.

### Theta MLE
Newton-Raphson on the Rasch score equation. Triggers once `--theta-min-items` (default 10) items with known b are in the DB.

In the 15-profile architecture, θ MLE is still computed for the layperson (profile 0, stored as "vanilla") and expert (profile 14, stored as "augmented") using the same DB response vectors. Too-hard items (P < 0.10) have large positive b values and enter the MLE vector as failures for both — this prevents augmented θ from diverging to +∞.

- Vanilla θ (profile 0): estimated at ≈ −0.74 (layperson fails most regulatory items)
- Augmented θ (profile 14): converges above 0 now that too-hard items provide finite negative MLE terms

### Single persistent SQLite connection
`Database` opens one connection in `__init__` and reuses it. The `with self._connect()` pattern returns the same connection — does NOT open/close per call.

### Schema migration
`Database._migrate()` uses `PRAGMA table_info` to add columns to existing DBs without dropping data. Used to add `source_type` and `source_ref` columns.

### Phase 1 baseline calibration (Rasch via girth)
`run_phase1.py` reads `configs/phase1_baseline.yaml`, builds a 45-cell virtual-examinee grid (3 models × 5 temps × 3 strictness levels), runs every (examinee × item) pair through `arena/runner.py`, fits Rasch (1PL) via `girth.rasch_mml`, and post-hoc shifts every b so the baseline examinee's MLE θ equals 0.

- **Baseline examinee** (config-driven): `(llama-3.1-8b-instant, t=0.1, strictness=none)`. Anchoring formula: pick c = θ_baseline_raw, then b_anchored = b − c so θ_baseline_anchored = 0 by definition.
- **Resumability**: the runner skips pairs whose `(examinee_id, task_id)` already appears in `responses.jsonl`. To re-run a single cell after a bug fix, delete its rows from the jsonl and rerun — the runner picks up only the missing pairs.
- **Outputs**: per-row jsonl in `logs/arena_runs/{run_id}/responses.jsonl`; per-item b in `evaluator/output/{run_id}_phase1_b.jsonl`.

### Phase 2 Rasch + classical QC (the official path)
`run_phase2.py` reads the Phase 1 `responses.jsonl`, refits Rasch via `girth.rasch_mml`, and applies three classical QC signals computed on the response matrix. Output: `evaluator/output/phase2_qc_report.json` and `evaluator/output/phase2_frozen_bank.jsonl` (schema: `{task_id, b, point_biserial, infit, outfit}`).

- **Why Rasch, not 2PL**: 2PL is not identifiable on our 45-cell grid. `twopl_mml` (marginal MLE) pinned every item at a=5.0 (girth's upper bound); `twopl_jml` (joint MLE) gave a real spread but JML is statistically inconsistent as n_items→∞ with n_examinees fixed, which is exactly our regime. Rasch is identifiable, gives specific objectivity for Δθ comparisons in Phase 3, and matches the model used in Phase 1.
- **QC signals** (see `evaluator/rasch_qc.py`):
  1. **Zero-variance**: items with P=0 or P=1 have undefined Rasch b at the bound. Drop unconditionally.
  2. **Point-biserial** (corrected: item vs total − item): negative pb means the item is inverted (smarter examinees fail more) — same broken-ground-truth signal that 2PL `a ≤ 0` was meant to catch. Default threshold `pb ≤ 0.0`.
  3. **Infit / Outfit** (Rasch fit MSRs computed using MLE θ): caught noisy items where the model doesn't fit. Asymmetric defaults (upper bound only): infit ≤ 1.5, outfit ≤ 2.0. Linacre's standard symmetric [0.5, 1.5] is calibrated for thousands of examinees — on 45 cells, "too deterministic" items (low MSR) are clean separators we want to keep, not defects.
- **Why outfit gets a looser upper bound (2.0 vs infit's 1.5)**: outfit is unweighted, so a single anomalous cell at extreme θ blows it up disproportionately on a small grid.
- **Examinee θ for fit stats**: MLE (matches Linacre/WINSTEPS practice). Non-finite θ examinees would be dropped from the residual computation; on the 45-cell grid all 45 are finite.
- **2PL machinery is dormant, not deleted**: `evaluator/twopl.py` and `evaluator/qc.py` are kept as a future path. They use `girth.twopl_jml` with `a_threshold=0.3` and were the active path 2026-05-10 morning before being demoted in favor of Rasch+classical-QC for the reasons above.

### Parameter uncertainty: analytic SE now, py-irt posterior SD later (2026-06-09)
`girth.rasch_mml` returns **point estimates only** — no standard errors. Two uncertainty paths exist:

- **Analytic SE (interim, zero API calls)**: `SE(b) = 1/√(N·P(1−P))` with N=45 and `P = σ(−b)` recovered from each frozen-bank b. Standard delta-method SE for a one-item logit difficulty. Implemented in `paper/figures/_common.py::analytic_se_b`; this is what the paper reports today.
- **py-irt posterior SD (intended final)**: `evaluator/pyirt_fit.py` fits the same 1PL via py-irt (Pyro BBVI) and exposes `loc_diff → b_mean`, `scale_diff → b_se` (per-item posterior SD). Baseline anchoring mirrors `evaluator/rasch.py` exactly (shift b by −θ_baseline; b_se invariant). **Blocked on data**: needs the raw `logs/arena_runs/phase1_baseline/responses.jsonl`, which is gitignored and absent in fresh clones — regenerate via `run_phase1.py` (~164k Groq calls) before py-irt can run on the real bank. Recovery verified on synthetic data (r≈0.93). py-irt b has a wider scale than girth (vague priors, 45 examinees) — wider posterior SDs are the honest uncertainty, not a defect.

### Phase 3a Δθ from prompt strictness (offline + lenient rejudge)
`run_phase3.py` measures how much system-prompt strictness moves a model's θ on the frozen Phase 2 bank. Pure offline analysis on Phase 1's existing `responses.jsonl` — no new solver calls. Output: `evaluator/output/phase3a_strictness_deltatheta.json`.

- **Examinee scope**: 3 models (llama-3.3-70b-versatile, openai/gpt-oss-20b, llama-3.1-8b-instant) × 1 temperature (default 0.4) × 3 strictness levels (none / neutral / strict) = 9 cells, plus the Phase 1 baseline (llama-3.1-8b @ t=0.1, none) as a θ=0 sanity anchor.
- **Subset re-anchor**: the Phase 2 bank's `b` was anchored on the *full* 1,823-item Phase 1 set; restricting to the 1,284 healthy items shifts the baseline's MLE θ by ≈ −0.72. `run_phase3.py` re-anchors *inside the script* (subtract baseline θ from every `b`) so absolute θ values are honest. Δθ comparisons within a model are invariant to this shift.
- **Format-vs-reasoning decomposition**: `run_phase3_rejudge_lenient.py` re-grades all 9 subject cells × 1,284 items with the lenient judge (no citation requirement) — 11,556 judge calls, resumable. Then `run_phase3.py` computes a parallel lenient Δθ table and the *format-unlock* component = strict Δθ − lenient Δθ.
  - **Why this matters**: the strict judge requires section-level CFR/ICH citations. A "system-prompt effect" can mean two very different things — (a) the prompt taught the model to *cite* (format unlock), or (b) the prompt taught the model to *reason* better (genuine compliance gain). The lenient pass strips (a) and reveals only (b).
- **2026-05-10 result** (see §7): the 70b's headline +5.41-logit prompt swing decomposes into ~+2.5 reasoning + ~+2.9 format. gpt-oss-20b's prompt effect is essentially all format (lenient Δθ ≈ 0). Honest takeaway: system-prompt strictness primarily teaches *citation format*, not regulatory reasoning.

### Phase 3b agent variants (2026-06-10)
`run_phase3b.py` tests whether *architectural* interventions — not prompt edits — move θ on the frozen Phase 2 bank. Three new agents in `agents/`, each a drop-in for ZeroShotAgent and each isolating one mechanism:

- **RetrievalAgent** (`agents/retrieval.py`): embeds `context + question`, retrieves top-k (default 3) CFR sections via `harness/cfr_store.py`, prepends them as a "Relevant Regulations" system-prompt block, then makes the same single solver call as zero-shot. Isolates *in-context regulatory grounding*.
- **CriticAgent** (`agents/critic.py`): initial answer → n_rounds self-critique ("review for citation accuracy and compliance determination") → final synthesis. 2 + n_rounds calls/item (default 3). Isolates *self-review*.
- **StepDecompositionAgent** (`agents/step_decomp.py`): 4 sequential calls — identify activity/parties → identify governing provision → assess compliance → final determination with citation. Isolates *explicit CoT scaffolding*.

**Backward compatibility (load-bearing):**
- `arena/grid.py::_make_examinee_id` omits `|a=zero_shot` from the hash string when `agent_type="zero_shot"`, so every Phase 1/3a examinee ID — and therefore every existing `responses.jsonl` row — is unchanged. Non-zero_shot agents get `|a={agent_type}` appended.
- `arena/schema.py::ArenaLogEntry.agent_type` defaults to `"zero_shot"`; old log rows parse without it.
- `arena/runner.py::ArenaRunner` takes an optional `agent_factory: Callable[[ExamineeConfig], Agent]`; when `None` it constructs ZeroShotAgent exactly as before, so Phase 1/2/3a entry points are untouched.

**CFR vector store**: `scripts/build_cfr_index.py` downloads eCFR XML (21 CFR Parts 11, 50, 58, 211 + 45 CFR 46), extracts section-level text, embeds with `all-MiniLM-L6-v2`, writes `data/cfr_index/{embeddings.npy, sections.jsonl}` (gitignored — rebuild after fresh clone). `CFRStore.retrieve()` uses pure-numpy cosine similarity (dot product of unit vectors); no FAISS dependency, <1 ms over ~1k sections.

**Grid & budget**: `configs/phase3b_variants.yaml` — 2 models (70b, 8b) × t=0.4 × 2 strictness (none/strict) × 4 agent_types = 16 cells × 1,284 frozen items. ~41k solver + ~20k judge Groq calls (critic 3×/item, step_decomp 4×/item). Resumable via the standard runner skip logic. Output: `evaluator/output/phase3b_agent_deltatheta.json` with per-cell θ and Δθ vs the same-(model, strictness) zero_shot cell.

### Reasoning-model handling in `harness/groq_client.py`
Two flavours of reasoning model on Groq, handled differently:

- **Inline-CoT models** (e.g. `deepseek-r1-distill-llama-70b`): emit `<think>...</think>` blocks inside `message.content`. Stripped by `_THINK_BLOCK_RE` so callers see the clean answer.
- **Separate-channel models** (`openai/gpt-oss-*`): emit chain-of-thought into `message.reasoning` and the answer into `message.content`. With the SDK default `reasoning_effort="medium"` the CoT regularly consumes the entire `max_tokens` budget, leaving `message.content` empty. Mitigated at two layers:
  1. `_uses_separate_reasoning_channel(model)` triggers auto-injection of `reasoning_effort="low"` for any matching model.
  2. If `message.content` is still empty after stripping, the (also stripped) `message.reasoning` is used as `text` so the judge has something to grade. Tracked in `raw_meta.used_reasoning_fallback`.

This was a 2026-05-09 fix for a bug where gpt-oss-20b @ strict produced 20% empty responses (1820 / 9115). Diagnostic showed default reasoning_effort caused `finish_reason='length'` with all 1024 tokens consumed by reasoning. Verified post-fix: 0/50 empties on a same-cell sample.

---

## 6. What to Avoid

### Don't use system Python
macOS Python 3.12 (Homebrew) is externally managed. Always `source venv/bin/activate` first.

### Don't generate easy items with strict judge
Easy domain items + strict judge = layperson always fails = no easy retained items = θ MLE can't converge for the low-ability profiles.

### Don't confuse `is_retained` with difficulty band
`is_retained = True` for both hard and easy items. Check `retention_reason` to distinguish.

### Don't mix profile-population b values with legacy band-sampled b values
Pre-refactor DB rows have `b` values sampled from Uniform bands. Post-refactor rows have `b = ln((1−P)/P)` from pass rate. Identifiable: old rows also have `a != 1.0` (2PL era) or may have `b` falling exactly at Uniform boundaries. Do not average them without acknowledging the different estimation methods.

### Experiment history (from git log)
- Exp 1: Lenient judge → Vanilla passed everything → no hard items
- Exp 2: Counterintuitive hints → improved hard item yield
- Exp 3: Strict citation requirement in judge → current design
- Exp 4: Section-level citation required + simplified promo hints
- Exp 5: Regulatory reference toolkit in augmented prompt → reduced too-hard rate
- Exp 6 (this refactor): 15-profile synthetic population + logit pass-rate b formula

---

## 7. Current State

### Phase 1 calibration (active workflow)
- 1,823 items from `compliance_bank.db` calibrated against a 45-cell grid (3 models × 5 temps × 3 strictness) → 82,035 graded attempts, baseline-anchored b in `evaluator/output/phase1_baseline_phase1_b.jsonl`
- Baseline raw θ = **−0.626** → anchored to 0; b shifted by +0.626  (post-fix; was −0.513 pre-fix)
- b summary: min=−5.374, median=+2.442, max=+6.626
- Real-vs-synthetic b gap = **+0.49** (real n=105 mean +2.73; synthetic n=1718 mean +2.25). Real items still skew harder.
- Item bands: 294 at P=0 (16%), 240 at 0<P<0.05 (13%), 1241 informative 0.05–0.95 (68%), 34 at 0.95<P<1 (2%), 14 at P=1 (1%). Phase 2 2PL QC drops the floor automatically.

### Phase 2 frozen bank (Rasch + classical QC)
- 1,823 items in → **1,284 items frozen** (70.4% retention) in `evaluator/output/phase2_frozen_bank.jsonl`
- Dropped: 308 zero-variance + 155 low-point-biserial (≤ 0, inverted) + 14 infit > 1.5 + 62 outfit > 2.0
- Healthy median: b=+1.65, point_biserial=+0.42, infit=0.87, outfit=0.66
- Bank schema: `{task_id, b, point_biserial, infit, outfit}` per item
- QC report: `evaluator/output/phase2_qc_report.json`
- Pre-history (2026-05-10 morning, now superseded): a 2PL-JML pass kept 1,374 items with a/b. Switched to Rasch+classical QC because 2PL on 45 examinees is methodologically strained (see Phase 2 section in §5).
- **2026-05-09 fix verified**: gpt-oss-20b @ strict empties dropped 1820 / 9115 (20%) → **0 / 9115 (0%)** after the `reasoning_effort="low"` + reasoning-fallback fix in `harness/groq_client.py`. The cell's mean P rose 0.334 → 0.424 (+0.09) once the fake failures were removed; non-reasoning cells unchanged. Baseline raw θ shifted from −0.513 to −0.626 because the corrected gpt-oss-20b strict cell is genuinely stronger than baseline.
- Phase 1 infra is resumable; partial runs are safe.

### Phase 3a Δθ from prompt strictness (frozen bank, t=0.4, dual-judge)
- Examinees: 9 subject cells (3 models × t=0.4 × 3 strictness) + Phase 1 baseline (8b @ t=0.1, none) re-anchored to θ=0 on the 1,284-item subset.
- Lenient rejudge: 11,556 lenient-judge calls on the same `raw_response` text already saved in Phase 1's log; `~4.3%` judge errors (gracefully mapped to FAIL per Rule 3). Output: `logs/arena_runs/phase3a_lenient/responses.jsonl` (gitignored).

| Model                       | strict-judge Δθ (none→strict) | lenient-judge Δθ | format-unlock | reasoning gain |
|-----------------------------|------------------------------:|----------------:|--------------:|----------------:|
| llama-3.3-70b-versatile     | **+5.41**                     | +2.52           | +2.89         | +2.52           |
| openai/gpt-oss-20b          | +0.66                         | −0.23           | +0.89         | ≈ 0             |
| llama-3.1-8b-instant        | +1.69                         | +0.18           | +1.52         | ≈ 0             |

Headline: under the strict (citation-requiring) judge, prompt strictness moves all three models. But the lenient judge — which scores conclusion + reasoning only — flattens that effect for everything except 70b. Implication: **system-prompt strictness primarily teaches the model to satisfy the citation-format requirement, not to reason better about compliance.** Only the 70b shows a non-trivial reasoning gain (≈ +2.5 logits) on top of the format-unlock effect.

### Phase 3b agent variants (complete, merged 2026-06-10)
- Full 16-cell run done; results in `evaluator/output/phase3b_agent_deltatheta.json`; Table 2 filled in `paper/sections/07b_phase3b.tex`.
- Key finding: architectural interventions (retrieval +1.07, critic +3.45, step_decomp +4.55) boost 70b θ under lenient scoring, but all three reduce θ under strict citation-requiring scoring — same pattern as Phase 3a.

### autoevolve.py — frozen bank fitness function (2026-06-10)
`autoevolve.py` now uses the Phase 2 frozen bank (`evaluator/output/phase2_frozen_bank.jsonl`) as its IRT fitness function instead of legacy DB b values.

**Changes made:**
- `_load_frozen_bank(path)` — loads `{task_id: b}` from frozen bank jsonl (mirrors `run_phase3b.py`)
- `sample_test_items_from_bank(db, bank_b, n, seed)` — stratified item sampling using frozen bank b values (bands A/B/C/D), item content from DB
- `build_failure_report(db, bank_b=None)` — when `bank_b` provided, filters to bank items and uses bank b values for band analysis
- `test_proposal(..., bank_b=None)` — resolves item difficulty from bank first, DB fallback
- `evolve(..., bank_path=None, dry_run=False)` — loads frozen bank at startup, passes `bank_b` through the loop; `dry_run` prints sampled items + b values without calling Groq
- History entries now include `bank_path` and `n_bank_items` fields for auditability
- `legacy_evaluator.py` — renamed from `evaluator.py` to avoid clash with `evaluator/` package; `calibrator.py` updated accordingly

**CLI:**
```bash
python3 autoevolve.py --bank evaluator/output/phase2_frozen_bank.jsonl --dry-run --iterations 1 --test-items 5
python3 autoevolve.py --iterations 10 --test-items 40 --min-delta 0.50  # uses frozen bank by default
```

**Fallback:** if the frozen bank is not found, legacy DB b values are used with a warning.

### Legacy bank pipeline (still produces items)
- generate → calibrate (15 profiles) → filter by pass rate → logit b → persist
- **High-Subtlety generation model**: all hard items require Red Herrings + subtle violations
- Hard domains (5): counterintuitive scenarios with mandatory Red Herring, strict judge
- Easy domains (5): obvious scenarios, lenient judge (unchanged)
- 10 targeted 21 CFR Part 11 sub-domains (Hybrid Systems + Legacy Validation)
- 15-profile synthetic population: 5 expertise tiers × 3 temperatures (0.1–0.8)
- b estimation: `b = ln((1−P)/P)` from empirical pass rate P, P clipped to [0.01, 0.99]
- Retention thresholds: P < 0.05 → discarded_too_hard, P > 0.95 → retained_easy, else retained
- DB backward compat: profile 0 (layperson) → vanilla_*, profile 14 (expert) → augmented_*
- `pass_rate` column now persisted in `calibration_results` (migrated automatically)
- Real-world track: `fda_importer.py` fetches FDA warning letters, extracts items, calibrates
- `source_type` tagging: `synthetic` vs `real` in DB
- Real vs synthetic b-distribution comparison (`print_comparison`)
- Theta MLE via Newton-Raphson for layperson (≈ −0.74) and expert profiles
- CLI: `--n`, `--easy-n`, `--db`, `--quiet`, `--theta-min-items`, `--compare`, `--gen-21cfr11`, `--recalibrate-all`
- Background build scripts: `build_bank.sh`, `build_real.sh`

### Known issues
- Legacy DB rows with `a != 1.0` from 2PL era; legacy rows also have band-sampled b, not logit b
- Legacy rows calibrated with old thresholds (0.10/0.90) have stale `is_retained` flags — use `--recalibrate-all` to fix
- Promo_review domain underrepresented in real items (OPDP letters use different URL structure)
- 15 solver calls + 15 judge calls = 30 API calls per task — significantly slower than the 4-call 2-profile design; mitigated by Groq's fast inference
- fda_importer.py still uses the 2-profile Calibrator interface (backward compat shim in place)
- Red Herring quality not graded — items are generated with the Red Herring mandate but no automated check verifies a Red Herring was actually embedded

### Judge calibration audit module (2026-06-12, complete)
`judge_audit/` compares the project's IRT-based strict/lenient judge against DeepEval G-Eval and RAGAS on 400 Phase 3b zero-shot responses (100 frozen-bank items × 4 cells). Full run complete: 1,600 scores, 93 errors excluded (5.8%, mostly transient connection timeouts — resumable).

- **Data**: 100 frozen-bank items × 4 zero-shot cells = 400 responses from `logs/arena_runs/phase3b_variants/responses.jsonl` (zero new solver calls)
- **Metrics**: `geval_strict` (citation-requiring rubric), `geval_lenient` (conclusion-only), `ragas_faithfulness` (groundedness to context), `ragas_answer_correctness` (weights=[1,0], no embeddings)
- **Backend**: `GroqDeepEvalLLM` (subclass of `DeepEvalBaseLLM`) + `make_ragas_llm` (langchain `ChatOpenAI` → `LangchainLLMWrapper`) — both pointed at Groq `llama-3.3-70b-versatile`; same grader family as project judge controls for model capability
- **Grader dependencies**: deepeval 4.0.6, ragas 0.4.3, langchain<1/langchain-community<0.4/langchain-openai<1 (ragas 0.4 requires langchain 0.3.x — langchain 1.x breaks the import path)
- **Analyses** (in `judge_audit/analysis.py`): AUC (Mann-Whitney U/n₁n₂) + point-biserial vs IRT binary verdict; per-cell mean score vs Rasch θ (Spearman); difficulty-confounded leniency slope (score on b within PASS/FAIL); disagreement mining (top-20 high-geval-but-FAIL + low-geval-but-PASS)
- **CLI**: `python3 run_judge_audit.py --smoke` / `--n-items N` / `--analyze-only`

**Full-run results (2026-06-12):**

| metric | n | AUC vs IRT | r_pb | rho_θ | slope\|PASS | slope\|FAIL |
|---|---|---|---|---|---|---|
| geval_strict | 400 | **0.838** | 0.570 | 1.000 | +0.015 | −0.019 |
| geval_lenient | 400 | 0.731 | 0.381 | 1.000 | −0.015 | −0.052 |
| ragas_faithfulness | 363 | 0.530 | 0.057 | −0.800 | −0.073 | −0.056 |
| ragas_answer_correctness | 344 | 0.693 | 0.315 | 0.800 | −0.003 | −0.017 |

**Interpretation:**
- `geval_strict` AUC=0.838 confirms the citation-requiring G-Eval rubric is the best external proxy for our IRT judge — both penalize missing section-level citations.
- `ragas_faithfulness` AUC≈0.530 (near chance) validates the design choice to include it as a *contrast* metric: a response can be perfectly grounded in the scenario context and still wrong on compliance — faithfulness is not correctness.
- `rho_θ=1.000` for both G-Eval metrics (n=4 cells only — not statistically meaningful alone) suggests G-Eval ranks the 4 zero-shot cells in the same ability order as Rasch θ.
- Difficulty slopes are near zero conditional on verdict for all metrics — none of the external tools confound item difficulty with response quality to a meaningful degree. IRT's separation of b and θ is not replicated but also not strongly violated by G-Eval.
- `ragas_faithfulness` negative `rho_θ=-0.800`: higher-ability cells (70b) may give more citation-heavy responses that diverge further from the flat context, paradoxically reducing faithfulness scores.

### Gaps
- No promotional material real items retained yet (0 in real bank)
- No human expert validation of item quality or gold standards
- No adaptive item selection loop
- No export to standard IRT formats (mirt, py-irt)

---

## 8. Honest Assessment (Updated 2026-04-13)

This section documents an objective, critical evaluation of what has been built and what it is not.

### Industry value: 3 / 10 as-is

**Real strengths:**
- Regulatory domains (21 CFR Part 11, GCP, GMP, informed consent) are genuine industry pain points
- FDA warning letter track grounds items in actual enforcement actions
- The pipeline architecture (generate → calibrate → filter → persist) is a reasonable production skeleton

**Critical gaps for industry use:**
- No SME validation: gold standards are LLM-generated and LLM-graded. Any incorrect gold standard poisons the calibration. Regulatory nuance (exceptions, carve-outs, jurisdiction differences) requires expert review.
- The "IRT-calibrated" claim cannot be made to compliance stakeholders — b values lack standard errors, item fit statistics, or cross-validation. This is not psychometric certification.
- 130 items across 5 domains is too sparse. Real compliance exams have 200–500 items per domain.
- No production engineering: local bash scripts, no API, no audit trail, no item security.

### Academic value: 4 / 10 as-is

**What is publishable:**
- The dual-track design (synthetic LLM-generated + real FDA enforcement letters) as a method for grounding automated benchmarks is a legitimate research idea
- The real-vs-synthetic b-distribution gap is now formally tested (Mann-Whitney U, two-sided p=0.0012, Cliff's δ=+0.20) — see §5 'Real-world track rationale'
- Strict/lenient judge design and its effect on vanilla pass rate is a practical engineering insight

**Critical methodological problems:**
- **15-profile IRT is better but still not psychometric IRT.** IRT calibration requires many independent test-takers (typically 200–1000+). 15 profiles all running the same base model with different system prompts and temperatures are not independent — they are highly correlated. b values from this procedure are more informative than 2-profile band sampling but lack standard errors or item fit statistics.
- **The logit b formula assumes profiles are centered at θ=0.** This is an untested assumption. If the 15 profiles systematically lean expert (mean θ > 0), b will be biased downward (items appear easier than they are).
- **Closed evaluation loop with no external validity.** Same model family generates, solves, and judges. No measurement of whether the difficulty ranking matches human expert judgments, real exam difficulty, or regulatory enforcement complexity.
- **Selection bias is reduced but not eliminated.** With 15 profiles, too-hard items (P < 0.10) are retained in the θ MLE vector, reducing augmented θ divergence. But items still must have P > 0 to enter the calibrated bank.

**Realistic publication target:**
- Workshop/short paper (not main venue) on automated benchmark construction in regulated domains, framed around the pipeline design and FDA grounding mechanism, with limitations section that explicitly names the above. A full venue paper now requires human expert grading of a stratified subset and item fit statistics (profile-to-profile consistency).

---

## 9. Next Steps (priority-ordered)

### Phase 1 → Phase 2 → Phase 3 roadmap
1. ~~**Finish Phase 1 rerun**~~ ✓ Done (2026-05-09) — gpt-oss-20b @ strict re-run with reasoning fix; baseline raw θ now −0.626 (anchored to 0).
2. ~~**Phase 2: Rasch + classical QC**~~ ✓ Done (2026-05-10) — `run_phase2.py` wired in. Rasch + (zero-variance, point-biserial, infit, outfit) QC on 1,823 items → 1,284 frozen items in `evaluator/output/phase2_frozen_bank.jsonl`. 2PL machinery preserved as dormant path for future re-evaluation when bank/examinee count grows.
3. ~~**Phase 3a: Δθ from prompt strictness (offline + dual-judge)**~~ ✓ Done (2026-05-10) — `run_phase3.py` + `run_phase3_rejudge_lenient.py`. Strict-judge Δθ shows large prompt effects (70b: +5.4 logits); lenient-judge re-grade reveals the format-unlock vs. reasoning decomposition: only the 70b shows non-trivial reasoning gain (~+2.5), while gpt-oss-20b and llama-3.1-8b have ≈ 0 lenient Δθ — i.e., system prompts mostly buy citation format, not regulatory reasoning. See §5 / §7.
4. **Phase 3b: agentic / harness changes — code complete (2026-06-10), run pending** — `RetrievalAgent` (CFR vector retrieval), `CriticAgent` (self-review), and `StepDecompositionAgent` (4-step scaffold) implemented on branch `phase3b-agent-variants` with backward-compatible arena plumbing; entry point `run_phase3b.py`. Remaining: build CFR index, smoke test, full 16-cell run (~41k solver calls), fill paper Table 2. See §5 / §7 "Phase 3b agent variants".
5. **Future: revisit 2PL** — once the bank is bigger and examinee count grows beyond the 45-cell grid, refit 2PL via `evaluator/twopl.py` (already implemented) and compare against Rasch.

### To make academic contribution credible
- ~~**Add solver profiles**~~ ✓ Done — 15 synthetic profiles spanning layperson → senior FDA auditor (legacy pipeline); 45-cell grid spans 3 base models × 5 temps × 3 strictness levels (Phase 1 pipeline)
- **Human expert review — IN PROGRESS (2026-06-11)** — 50-item stratified sample exported via `scripts/export_expert_sample.py` to `expert_review/` (reviewer CSV + held-back answer key + instructions). Sample: 10 per hard domain, QC-filtered (pb ≥ 0.3, infit ≤ 1.3, outfit ≤ 1.5), b in P20–P80 band [0.83, 2.43], seed=42. Verdict balance: 38 VIOLATION / 12 COMPLIANT — a true 25/25 split is impossible because the High-Subtlety template makes hard items violation-by-design (only 16 COMPLIANT-verdict items exist in the entire QC'd hard-domain pool; all are band-exempt and 12 fit under the 10-per-domain cap). **Design lesson for bank regeneration: mandate ~50% compliant-scenario hard items.** Awaiting labels from 2–3 compliance professionals; two-step protocol (own verdict first, then grade the gold standard) yields both inter-rater agreement and gold-standard error rate.
- ~~**Mann-Whitney U test**~~ ✓ Done (2026-07-29) — `scripts/real_vs_synthetic_test.py`, stdlib-only, reproducible from the committed frozen bank
- **Cross-model profiles** — Phase 1 already includes 3 model families; expand if Phase 2 QC reveals high within-family correlation

### To improve current pipeline
5. **Promo_review real items** — OPDP warning letters are at a different URL pattern; need targeted search
6. ~~**Re-calibrate existing DB items**~~ ✓ Implemented — run `python3 main.py --recalibrate-all` to replace all stale rows with 15-profile logit b values and 0.05/0.95 thresholds
7. **Red Herring audit** — sample 20 items and manually verify that a genuine Red Herring is present and correctly described in the gold standard

### Longer term
- Adaptive item selection: given solver θ, pick item with b closest to θ (maximum Fisher information)
- Export to R `mirt` or Python `py-irt` for proper psychometric analysis
- SME item review workflow before any claim of industry use

# BIO_Compliance_IRT

Automated bio-compliance IRT item bank builder and LLM evaluation framework. Generates
biopharma regulatory compliance scenarios, calibrates item difficulty with Rasch IRT, and
uses calibrated ability θ as an optimization metric for evaluating LLMs.

## What it does

Two parallel pipelines share a SQLite item bank:

1. **Legacy bank pipeline** — generates compliance items (15 synthetic solver profiles,
   pass-rate → logit-b), persists to `compliance_bank.db`. Also ingests real FDA warning
   letters via `fda_importer.py`.
2. **Phase 1–3 calibration pipeline** — reads `compliance_bank.db` (read-only), runs a
   45-cell virtual-examinee grid (3 models × 5 temperatures × 3 strictness levels), fits
   Rasch (1PL) via `girth`, applies classical item-fit QC, and freezes a healthy bank.
   Phase 3 measures Δθ from system-prompt strictness, decomposing it into format-unlock
   vs. genuine reasoning gain.

## Quick start

**Requirements:** Python 3.10+, Groq API key.

```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Set Groq API key
export GROQ_API_KEY=gsk_...
# or: echo "GROQ_API_KEY=gsk_..." > .env && export $(cat .env | xargs)

# Legacy pipeline — generate 5 hard + 2 easy tasks
python3 main.py --n 5 --easy-n 2

# Phase 1 calibration — full run from configs/phase1_baseline.yaml (~164k Groq calls)
python3 run_phase1.py

# Phase 1 smoke test (3 items × 45 examinees)
python3 run_phase1.py --smoke

# Phase 2 QC — freeze the bank
python3 run_phase2.py

# Phase 3a — Δθ from prompt strictness (offline, no new solver calls)
python3 run_phase3.py
```

## Project structure

```
# Legacy bank pipeline (item generator)
main.py                         — CLI entry point
task_generator.py               — LLM generates compliance scenarios (5 hard + 5 easy domains)
calibrator.py                   — 15 solver profiles + judge → pass rate → logit b
evaluator.py                    — LLM-as-judge (strict and lenient modes)
irt_parameters.py               — Rasch math, outcome classification, theta MLE
database.py                     — SQLite persistence
fda_importer.py                 — Real FDA warning letter ingestion
build_bank.sh / build_real.sh   — Background build scripts

# Phase 1–3 calibration pipeline
run_phase1.py                   — Phase 1: 45-cell grid → Rasch fit + anchor
run_phase2.py                   — Phase 2: Rasch refit + classical QC → frozen bank
run_phase3.py                   — Phase 3a: offline Δθ analysis
run_phase3_rejudge_lenient.py   — Phase 3a: re-grade responses with lenient judge
configs/phase1_baseline.yaml    — 45-cell grid config
agents/                         — ZeroShotAgent
arena/                          — grid generator, item loader, resumable runner
evaluator/                      — Rasch fit, Phase 2 QC, response matrix
harness/                        — GroqClient, Judge

# Paper scaffold
paper/main.tex                  — arXiv preprint root
paper/sections/                 — per-section .tex files
paper/figures/                  — matplotlib figure scripts + generated PDFs
paper/references.bib            — BibTeX entries

# Key outputs
compliance_bank.db                                    — shared SQLite item bank
evaluator/output/phase2_frozen_bank.jsonl             — 1,284 healthy items (b, pb, infit, outfit)
evaluator/output/phase3a_strictness_deltatheta.json   — Δθ tables (strict + lenient)
```

## Tech stack

- **Python 3.10+** with `venv`
- **Groq API** (`https://api.groq.com/openai/v1`) via the `openai` SDK — model `llama-3.3-70b-versatile`
- **girth** — Rasch (1PL) MML fit
- **py-irt** — variational Bayes 1PL with posterior SDs (requires regenerating `responses.jsonl`)
- **SQLite** (stdlib) — no ORM

## Current bank (Phase 2 frozen)

| | count |
|---|---|
| Input items (Phase 1) | 1,823 |
| Dropped: zero-variance | 308 |
| Dropped: low point-biserial | 155 |
| Dropped: infit > 1.5 | 14 |
| Dropped: outfit > 2.0 | 62 |
| **Frozen (healthy)** | **1,284** |

Healthy-bank medians: b = +1.65, point-biserial = +0.42, infit = 0.87, outfit = 0.66.

## License

MIT

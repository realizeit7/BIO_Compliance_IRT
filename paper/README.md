# Paper: Calibrated Benchmarking of LLMs in Regulated Domains

arXiv preprint scaffold for the BIO-Comply IRT project. Two-part structure:

- **Part 1 (the instrument)** — Sections 3–6: automated item generation,
  synthetic-population calibration, the Rasch pipeline, and the frozen bank.
- **Part 2 (the experiment)** — Section 7: using calibrated ability θ as an
  optimization metric, with the strict/lenient format-vs-reasoning decomposition.

## Layout

```
paper/
  main.tex             document root
  references.bib       BibTeX stubs (verify before submission)
  sections/00..09      one .tex per section
  figures/
    _common.py         shared loaders + analytic SE
    gen_fig1..4_*.py   matplotlib scripts (read existing pipeline outputs)
    fig1..4_*.pdf/png  generated figures
```

## Build

Figures (no LaTeX needed; matplotlib outputs PDF directly):

```bash
source ../venv/bin/activate
cd figures
python3 gen_fig1_b_hist.py
python3 gen_fig2_item_fit.py
python3 gen_fig3_delta_theta.py
python3 gen_fig4_decompose.py
```

PDF (requires a LaTeX toolchain — not installed in the web container):

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Data sources

All figures and tables read from existing pipeline outputs — **no API calls**:

- `../evaluator/output/phase2_frozen_bank.jsonl` — 1,284 healthy items (b, pb, infit, outfit)
- `../evaluator/output/phase3a_strictness_deltatheta.json` — Δθ tables

## Uncertainty (b ± SE)

The draft reports an **interim analytic SE** = `1/sqrt(N·P(1−P))` (N=45),
computed in `figures/_common.py::analytic_se_b`. The intended final
uncertainty is the **py-irt posterior SD**, implemented in
`../evaluator/pyirt_fit.py`. That path needs the raw response matrix
(`../logs/arena_runs/phase1_baseline/responses.jsonl`), which is gitignored
and must be regenerated:

```bash
python3 ../run_phase1.py    # ~164k Groq calls, resumable — rebuilds responses.jsonl
python3 ../run_phase2.py    # re-freeze the bank from the regenerated matrix
# then call evaluator/pyirt_fit.fit_pyirt(...) to fill b ± SE
```

Search the `.tex` files for `\todo{...}` (17 placeholders) for the prose and
numbers still to fill in.

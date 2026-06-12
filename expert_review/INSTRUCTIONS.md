# Expert Review Instructions — Bio-Compliance Item Validation (v1)

Thank you for reviewing these items. This set contains **50 regulatory
compliance scenarios** spanning five domains: 21 CFR Part 11 (electronic
records/signatures), GCP deviations, GMP deviations, informed consent, and
promotional review. Some are derived from actual FDA warning letters; most are
synthetic scenarios. All questions and reference answers were drafted by an AI
system — **your job is to tell us where the AI got it wrong.**

Estimated time: ~5 minutes per item, ~4 hours total. Partial completion is
still valuable — please work in order from item 1.

## Protocol (per item — please follow the order)

Open `expert_review_sample_v1.csv` (Excel/Google Sheets friendly).

**Step 1 — Your verdict first (before reading the reference answer).**
Read only `context` and `question`. In the column
`expert_verdict (VIOLATION/COMPLIANT)`, enter:
- `VIOLATION` — the scenario contains a regulatory compliance gap
- `COMPLIANT` — the described process/decision is acceptable as presented

**Step 2 — Grade the reference answer.**
Now read the `gold_standard` column. In `gold_standard_correct (Y/N)`, enter:
- `Y` — the reference answer reaches the right conclusion AND cites the right
  regulation(s) at the section level
- `N` — wrong conclusion, wrong/missing citation, or materially misleading
  reasoning

**Step 3 — Comments (required when you enter N, optional otherwise).**
Briefly state what is wrong: e.g. "cites 11.10(e) but the operative section is
11.10(k)", "the 'violation' described is actually permissible under the
predicate rule", "scenario is ambiguous — both verdicts defensible".

## Notes

- Many scenarios deliberately include a *red herring* — a process that looks
  suspicious but is actually compliant. Judge the scenario as a whole.
- If a scenario is too ambiguous to call, enter `AMBIGUOUS` in the verdict
  column and explain in comments — that itself is a finding we need.
- Please do not consult AI tools while reviewing; we are measuring human
  expert judgment.
- Citations standard: FDA regulations in 21 CFR (Parts 11, 50, 56, 58, 211)
  and 45 CFR 46; ICH E6(R2) for GCP.

## What happens with your labels

Your verdicts are compared against the AI reference answers to estimate the
error rate of the AI-generated answer key, identify domains needing
regeneration, and anchor the item difficulty scale to human expert judgment.
You will be acknowledged in any resulting publication (or kept anonymous —
your choice).

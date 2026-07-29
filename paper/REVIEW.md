# Review — *Calibrated Benchmarking of LLMs in Regulated Domains* (draft of 2026-07-29)

Reviewed against the repo at commit `86825fb`. Every number in the paper was
recomputed from committed artifacts where possible. Section/line references are to
`paper/sections/*.tex`, which matches the compiled PDF verbatim.

**Overall.** The instrument is real, the writing is unusually candid about its own
limits, and the format-vs-reasoning decomposition is a genuine result. Almost every
reported number reproduces exactly. But there is one factual error about the
experimental design in Section 8 that invalidates its stated mechanism, and three
properties of the difficulty scale that the paper does not disclose and that a
psychometrics-literate reader will find immediately. These are fixable — mostly by
rewriting claims to match what the code actually does — but they are not cosmetic.

---

## A. Numbers that verify

Recomputed from `evaluator/output/` and `compliance_bank.db`:

| Claim | Paper | Recomputed | |
|---|---|---|---|
| QC funnel 1,823 → 1,284 (308/155/14/62) | ✓ | `phase2_qc_report.json` | ✓ |
| Graded attempts 82,035 | ✓ | 1,823 × 45 | ✓ |
| Median b / pb / infit / outfit | +1.65 / +0.42 / 0.87 / 0.66 | +1.650 / +0.415 / 0.872 / 0.662 | ✓ |
| Real vs synthetic mean b | +1.98 / +1.04, gap +0.94 | +1.975 / +1.038, gap +0.938 | ✓ |
| Mann–Whitney U, z, p, δ | 66,516 / −3.24 / 0.0012 / +0.20 | 66,516 / −3.237 / 0.00121 / +0.201 | ✓ |
| b<0: 5/93 real, 372/1,191 synthetic | 5.4% / 31.2% | identical | ✓ |
| Easiest real −2.00, easiest synthetic −4.25 | ✓ | −1.999 / −4.250 | ✓ |
| Table 5 (Phase 3a Δθ) all cells | ✓ | `phase3a_strictness_deltatheta.json` | ✓ |
| Table 6 (Phase 3b Δθ) all 6 rows | ✓ | `phase3b_agent_deltatheta.json` | ✓ |
| Table 7 (judge audit AUC, r_pb, n) | ✓ | `judge_audit_report.json` | ✓ |
| Judge-audit errors 93/1,600 = 5.8% | ✓ | 400+400+344+363 = 1,507 | ✓ |

Arithmetic, funnels, and effect sizes are sound. The self-criticism in §6.2 (clustering),
§8.3 (unanchored scale), and §11 is accurate and well judged.

---

## B. Major — must fix before submission

### B1. Section 8 describes the wrong independent variable

**`strictness` in the Phase 3b grid is the solver's system prompt. It is not the judge.**

- `agents/zero_shot.py:26-38` — `STRICTNESS_LEVELS` maps `none`/`neutral`/`strict` to
  three *solver* system prompts.
- `arena/runner.py:205-206` — `_select_judge(item)` returns the lenient judge **iff**
  `item.domain.startswith("easy_")`. Judge mode is a property of the *item*, never of
  the examinee.
- `run_phase3b.py:201-214` — every one of the 16 cells receives the same
  `judge_strict` + `judge_lenient` pair.

The judge is therefore **identical across all 16 Phase 3b cells**. There is no lenient-judge
condition in Section 8 at all. The following statements are wrong:

- `07b_phase3b.tex:75-77` (Table 6 caption): "raise θ under lenient scoring but lower it
  under strict, citation-requiring scoring"
- `07b_phase3b.tex:95-97`: "depending on whether the judge requires a citation. Under
  lenient scoring (strictness=none)…"
- `07b_phase3b.tex:120`: "Retrieval gains +0.35 logits under lenient scoring"
- `09_conclusion.tex:22-24`: "All three raised θ under lenient scoring and lowered it
  under strict scoring"

The **result** survives; the **mechanism** does not. The correct reading is: *when the
solver is given no system prompt, agent scaffolds help; when the solver already has a
citation-demanding system prompt, they hurt.* That is arguably a better finding — the
scaffolds are substituting for the prompt, not for the grader — but it is a different
claim and Section 8 currently cannot support the one it makes.

Note this also breaks the parallel drawn to Section 7. Section 7's strict/lenient contrast
is a genuine *judge* manipulation on fixed response text. Section 8's is a *solver prompt*
manipulation with fresh responses. They are not the same comparison and should not be
described with the same vocabulary.

### B2. The +4.55 headline is measured from a 7% floor

Table 6 reports Δθ without the reference cell's base rate. From `phase3b_agent_deltatheta.json`:

| model / strictness | zero-shot pass rate | θ |
|---|---|---|
| 70b, none | **0.070** | −3.28 |
| 70b, strict | 0.578 | +1.90 |
| 8b, none | 0.268 | −0.72 |
| 8b, strict | 0.443 | +0.90 |

The 70b with no system prompt passes 7% of the bank — it is on the floor, because it does
not volunteer section citations. Every "gain" in the `strictness=none` column is measured
against that floor, where the logit scale is steepest and least stable. Step decomposition's
+4.55 moves the pass rate 0.070 → 0.491; the same intervention under a strict prompt moves
0.578 → 0.506. Reporting +4.55 next to −0.51 without the base rates invites the reader to
treat them as commensurable effects. **Add a zero-shot reference column (θ and pass rate)
to Table 6.**

### B3. The lower half of the difficulty scale is a judge artifact

23.1% of the frozen bank (297 items) comes from `easy_*` domains and is graded by the
**lenient** judge. The separation is total:

| | n | mean b | median | min | max |
|---|---|---|---|---|---|
| lenient-judged (`easy_*`) | 297 | −2.12 | −2.20 | −4.25 | **−0.83** |
| strict-judged (hard/real) | 987 | +2.07 | +2.20 | −3.50 | +4.25 |

Every lenient-judged item sits at b ≤ −0.83. **79% of all items with b < 0, and 90% of items
with b < −1, are lenient-graded.** So the "difficulty spans roughly ±4 logits" claim (§6.1,
abstract, conclusion) describes the span of a *two-rubric design*, not variation in item
content difficulty. Rasch assumes one measurement operation across items; here the rubric
predicts an item's position on the scale almost perfectly.

§5.2 justifies the lenient judge for easy items ("without it the easy items would be failed
by weak examinees… which would collapse the low-difficulty anchor") — but that is precisely
the admission that the anchor is constructed by the grader. This needs to be stated as a
limitation, and the ±4-logit span should be qualified. Infit/outfit will not catch it: the
easy items separate examinees consistently, so they look like clean items.

### B4. The scale has 44 distinct values, and 10% of items sit on the boundary

With 45 examinees, Rasch `b` is a function of the item's raw score (0–45), so at most 46
values are attainable. The 1,284-item bank resolves to **44 distinct b values**; minimum gap
between adjacent values is 0.108 logits; the ten largest score groups contain 61% of the bank;
and **128 items (10.0%) share the single value b = +4.249**, which is the estimator boundary.

Figure 2 is drawn as a continuous histogram and reads as a smooth difficulty distribution.
It is a 44-point lattice with a boundary spike. Consequences to state:

- "per-item difficulty" is really per-score-group difficulty; 1,284 items provide 44
  distinguishable levels.
- The 128 boundary items are censored, not estimated, and should be flagged as such.
- The adaptive-item-selection future work (§12) has 44 targets, not 1,284.

Suggested fix: plot as a step/stem chart, or overlay the lattice, and add one sentence
giving the 44/46 resolution bound.

### B5. "I report difficulty as b ± SE" — but no SE appears anywhere

`06_results.tex:51-52` promises `b ± SE`. No table, figure, or appendix in the paper reports
one. Table 4 gives a median b with no uncertainty; Figure 2 has no error bars;
`paper/figures/_common.py` defines `analytic_se_b` but no figure script calls it.

Computing the promised quantity (SE = 1/√(45·P(1−P)), P = σ(−b)) over the frozen bank:

- median SE = **0.498 logits**
- 42.4% of items have SE > 0.5; 18.2% > 0.75; **10.0% > 1.0**
- items at the +4.249 boundary have SE = **1.266**
- the range over which SE ≤ 0.5 is only **b ∈ [−2.20, +2.20]**

So roughly half of the advertised ±4-logit span carries per-item uncertainty of half a logit
or more, and the top decile is essentially uninformative. Either report the SEs (a second
panel on Figure 2, or a b-vs-SE scatter) or drop the sentence. Reporting them is the stronger
choice — it is consistent with the paper's tone, and §11 already prepares the reader for it.

### B6. No uncertainty on any Δθ, and the repo contains a replicate that supplies one

Every Δθ in Tables 5 and 6 is reported to two decimals with no interval. Two independent
sources of uncertainty are available and neither is used.

**(a) Model-based.** Rasch θ SE = 1/√(Σᵢ Pᵢ(1−Pᵢ)) on the frozen bank gives 0.074–0.123
per cell, so SE(Δθ) ≈ 0.11–0.15 logits.

**(b) Empirical — this is the more useful one.** Four zero-shot cells were administered
*twice*: once in the Phase 1 log, once again in the Phase 3b run. Same `examinee_id`, same
1,284 items, independent generations. Comparing raw (pre-anchor) θ:

| examinee | P (Phase 1) | P (Phase 3b) | raw θ₁ | raw θ₂ | \|Δθ\| |
|---|---|---|---|---|---|
| 8b, t0.4, none | 0.2749 | 0.2679 | −0.642 | −0.716 | 0.074 |
| 8b, t0.4, strict | 0.4618 | 0.4431 | +1.050 | +0.904 | 0.146 |
| 70b, t0.4, none | 0.0561 | 0.0701 | −3.576 | −3.279 | **0.297** |
| 70b, t0.4, strict | 0.5693 | 0.5779 | +1.836 | +1.896 | 0.060 |

**Mean |Δθ| = 0.144, max = 0.297 logits** from re-running the identical configuration.
Note this *exceeds* the model-based SE — local independence understates real reproducibility
variance, because each pair is administered once (`arena/runner.py`) and solver sampling
noise is not modelled.

This matters for specific claims:

- Table 5 lenient Δθ for gpt-oss-20b (**−0.23**) and llama-3.1-8b (**+0.18**) are *smaller
  than the demonstrated replicate noise*. The paper's "≈ 0" verdict is right, but it is
  currently asserted as a measured finding rather than as "indistinguishable from noise."
  Framing it the second way is both more honest and more persuasive.
- gpt-oss-20b's strict Δθ (+0.66) is only ~2× the noise floor and is treated as a real effect.
- Table 6's 8b rows (+0.09, −0.03) and the 70b retrieval-strict row (−0.30) are at or below
  the noise floor and should not be interpreted directionally.

**Recommendation:** report Δθ ± SE, use the replicate table above as an empirical noise
floor, and grey out or explicitly mark cells below it. This costs nothing — the data is
already committed — and it converts B6 from a weakness into a methodological strength.

---

## C. Moderate — correctness and reporting errors

### C1. Sign error in the anchoring identity (`05_irt_pipeline.tex:36-38`)

> "In the 1PL model P(correct|θ,b) = σ(θ−b), so shifting every b by a constant c shifts
> every θ by **−c**."

The Rasch likelihood is invariant under (θ, b) → (θ+c, b+c); θ and b shift **together**.
Should read "+c". The procedure that follows is correct (`evaluator/rasch.py` subtracts
θ̂_base from every b, so both shift down by θ̂_base and the baseline lands at 0) — only the
stated identity is wrong. In an IRT paper this is the kind of slip a reviewer will seize on.

### C2. §9.2 assigns the steepest slope to the wrong verdict class

> "The remaining metrics show small but significant negative slopes within **fail**, the
> steepest being RAGAS faithfulness at −0.073 (p < 10⁻⁷)"

From `judge_audit_report.json`, RAGAS faithfulness slopes are:
- within **PASS**: −0.0726, p = 6.3×10⁻⁸ ← this is the −0.073 quoted
- within **FAIL**: −0.0561, p = 1.1×10⁻⁶

The −0.073 is the within-PASS slope, so the gloss that follows ("failing responses to
harder items are scored slightly lower still") does not describe it. Also unmentioned:
G-Eval **lenient** has a significant within-PASS slope (−0.0150, p = 0.017), which is the
one metric-slope result that bears directly on the Section 7 decomposition, since G-Eval
lenient is the external analogue of the lenient judge.

### C3. §9.2 "one-in-twelve chance of perfect agreement" (`07c_judge_audit.tex:109-110`)

For n = 4 cells there are 4! = 24 orderings, so P(exact agreement) = **1/24 ≈ 4%**. 1/12 is
P(|ρ| = 1), i.e. perfect agreement *or* perfect inversion. Since the sentence is defending
a ρ = +1.00 result, 1/24 is the right figure — and it makes the point more strongly.

### C4. §9.2 AUC described as an agreement rate (`07c_judge_audit.tex:73-74`)

> "agree on which responses pass roughly five times out of six"

AUC = 0.838 is P(a random pass outscores a random fail). It is not classification agreement
and implies no accuracy figure. Reword: *"a passing response outscores a failing one about
five times in six."*

### C5. The ρ = 1.00 for G-Eval lenient is riding on noise

Cell means from `judge_audit_report.json`, `geval_lenient`: 0.625, 0.640, 0.645, 0.650 —
a **0.025 spread across 5.2 logits of θ**. The metric is effectively flat; the perfect
ordering is decided by differences of 0.005. G-Eval strict is genuinely responsive
(0.335 → 0.674) and deserves the claim; lenient does not. §9.2 hedges the pair jointly
("descriptive rather than inferential") but should separate them and give the ranges.
(The stored `spearman_p: 0.0` is also a degenerate scipy artifact for n = 4 — correctly
not quoted in the paper, but worth not trusting in the JSON either.)

### C6. Table 5's "reasoning gain" column is a definitional duplicate

Eq. (2) *defines* reasoning gain = Δθ_lenient. So columns 2 and 4 of Table 5 are the same
quantity, except that two real values (−0.23, +0.18) are replaced by "≈0" in column 4 —
which hides data and makes the table look like it contains more information than it does.
Drop the column and put the ≈0 judgement in prose (ideally with the noise floor from B6).

Relatedly, Table 5's columns do not add: 1.69 − 0.18 = 1.51, but format unlock is printed
as +1.52. The unrounded values (1.6915, 0.1760, 1.5155) are consistent; add a footnote that
columns are rounded independently.

### C7. Expert-review sampling frame is misstated (`07d_expert_review.tex:15-20`)

The paper says items were "restricted to… the interquintile difficulty band b ∈ [0.83, 2.43]".
`scripts/export_expert_sample.py:167-174` exempts COMPLIANT-verdict items from the band
entirely:

```python
# COMPLIANT-verdict items are structurally rare … so they are exempt
# from the difficulty band — we take any QC-passing COMPLIANT item.
band = [x for x in pool
        if x["verdict"] == "COMPLIANT"
        or (b_lo <= x["b"] <= b_hi and x["verdict"] == "VIOLATION")]
```

So **12 of the 50 items (24%) are outside the stated band**, and verdict class is confounded
with difficulty in the expert sample. Since one of the protocol's three outputs is
per-domain human/machine concordance, that confound is load-bearing and must be disclosed.

Also undisclosed: the VIOLATION/COMPLIANT labels are assigned by **regex over the
gold-standard prose** (`classify_verdict`, lines 47-68), not by a model or a human. The
38/12 balance is therefore itself an estimate. Say so.

### C8. "Judge error is real and measured" — but the measurement is withheld

`08_discussion.tex:60-63` states the rate exists without giving it. `CLAUDE.md` §7 records
**≈4.3% judge errors on the 11,556-call lenient re-grade**, all mapped to FAIL. This is not
a neutral omission: errors→FAIL biases the *lenient* θ estimates downward, and
format unlock = Δθ_strict − Δθ_lenient is therefore biased **upward** — in the direction of
the paper's headline conclusion. State the 4.3%, and state whether it was balanced across
the nine cells. If it was not, the format-unlock estimates need a correction or a caveat.

### C9. Code-availability claim is stronger than what ships

`10_availability.tex` says the Δθ tables are committed "so every number in this paper can be
recomputed without an API key." The Δθ tables are committed as **results**; their inputs
(`logs/arena_runs/*/responses.jsonl`) are gitignored, and `gen_fig3`/`gen_fig4` re-plot the
JSON rather than recomputing it. Genuinely recomputable without an API key: Section 6, the
Mann–Whitney test, Figures 1–2. Everything in Sections 7–9 is *verifiable* against committed
outputs but not *reproducible*. Reword to "recomputed or verified against committed outputs,"
and name which is which.

### C10. §3.2 "Five parallel easy domains"

Five easy domain keys are defined (`task_generator.py:98-143`), but the DB only ever contained
two: `easy_training` (254) and `easy_consent` (131). `easy_batch_record`, `easy_irb`, and
`easy_backup` produced zero items. The 297 easy items in the frozen bank come from two
domains (213 + 84). Given B3, the composition of the low end of the scale matters —
correct this.

---

## D. Minor

- **D1.** §5.1 says each pair is administered once. Given B6, add one sentence noting that
  single administration means solver sampling noise is not separable from ability, and point
  at the replicate as the empirical bound.
- **D2.** Related work omits IRT-for-AI work outside NLP benchmarks (Martínez-Plumed et al.)
  and holistic-evaluation framing (HELM). Also "Lalor et al. [6] **first** fit IRT to NLP test
  items" is a primacy claim — soften to "among the first" unless you want to defend it.
- **D3.** Table 4 reports only medians for a bank whose b distribution is left-skewed
  (mean +1.105 vs median +1.650). Give both, or an IQR.
- **D4.** §6.2's real/synthetic comparison inherits B3: real items are all strict-judged and
  71% of the synthetic bank is too, but the 297 lenient-judged synthetic items sit entirely
  in the low tail and pull the synthetic mean down. The +0.94 gap is partly a judge-mix
  artifact. Report the gap restricted to strict-judged synthetic items as a robustness check —
  it is a five-line computation and it either strengthens or usefully qualifies the result.
- **D5.** §8.3's "note on the scale" is good and correctly placed. Consider also stating that
  Δθ values in Sections 7 and 8 *are* mutually comparable as differences (same frozen b), which
  is not obvious to the reader after being told the scales differ.

---

## E. Suggested priority

1. Rewrite §8.5, the Table 6 caption, and the §12 conclusion sentence to describe the
   solver-prompt manipulation (B1). Add the zero-shot reference column (B2).
2. Add a paragraph in §5.2 or §6.1 disclosing the judge/difficulty confound, with the
   −0.83 separation figure (B3).
3. Add the 44-value resolution bound and the boundary pile-up to §6.1; redraw Figure 2 (B4).
4. Either report the analytic SEs or delete the promise (B5).
5. Add Δθ ± SE and the replicate noise floor (B6) — highest value per unit of effort, since
   the data is already in the repo.
6. Fix C1–C4 and C7–C8; they are small edits with disproportionate credibility cost.

The paper's greatest asset is that it argues against its own headline numbers more honestly
than most work in this area. Items B3–B6 are all in that same spirit — they are things the
author would clearly *want* to disclose, and disclosing them makes the central claim
(citation format dominates the measurement scale) harder to dismiss, not easier.

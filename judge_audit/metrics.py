"""Score audit rows with the four external metrics. Resumable: a
(metric, examinee_id, task_id) triple already present in the output jsonl is
skipped, mirroring arena/runner.py.

Metrics:
- geval_strict   DeepEval G-Eval, citation-requiring rubric (≈ our strict judge)
- geval_lenient  DeepEval G-Eval, conclusion-only rubric    (≈ our lenient judge)
- ragas_faithfulness        response grounded in scenario context (NOT correctness)
- ragas_answer_correctness  factuality vs gold standard (weights=[1,0]: no embeddings)

All metrics run on the same Groq grader model as the project judge
(see judge_audit/backends.py for why).
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from judge_audit.backends import DEFAULT_GRADER_MODEL, GroqDeepEvalLLM, make_ragas_llm
from judge_audit.sample import AuditRow

METRICS = ("geval_strict", "geval_lenient", "ragas_faithfulness", "ragas_answer_correctness")

# Explicit evaluation steps (instead of `criteria`) so G-Eval does not spend an
# LLM call auto-generating steps, and so the rubric is reviewable/deterministic.
_GEVAL_STRICT_STEPS = [
    "Compare the actual output's compliance determination (violation vs compliant) against the expected output.",
    "Check whether the actual output cites the specific regulatory section(s) named in the expected output (e.g. '21 CFR 11.10(e)'). Section-level citation is REQUIRED for a high score.",
    "Check whether the reasoning identifies the same core compliance issue as the expected output.",
    "Penalize heavily if the conclusion is wrong or the required section-level citation is missing or incorrect; a fluent answer without correct citations must not score high.",
]
_GEVAL_LENIENT_STEPS = [
    "Compare the actual output's compliance determination (violation vs compliant) against the expected output.",
    "Check whether the reasoning identifies the same core compliance issue as the expected output.",
    "Regulatory citations are NOT required; do not penalize their absence.",
    "Penalize only if the conclusion is wrong or the reasoning misses the core issue.",
]


def _make_geval(steps: list[str], name: str, grader_model: str):
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    return GEval(
        name=name,
        evaluation_steps=steps,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=GroqDeepEvalLLM(model=grader_model),
        async_mode=False,
        verbose_mode=False,
    )


def _score_geval(row: AuditRow, steps: list[str], name: str, grader_model: str) -> tuple[float, str]:
    from deepeval.test_case import LLMTestCase

    metric = _make_geval(steps, name, grader_model)
    tc = LLMTestCase(
        input=f"{row.context}\n\n{row.question}",
        actual_output=row.raw_response,
        expected_output=row.gold_standard,
    )
    metric.measure(tc, _show_indicator=False)
    return float(metric.score), str(metric.reason or "")


_RAGAS_LLM_LOCK = threading.Lock()
_RAGAS_LLM: dict[str, object] = {}


def _ragas_llm(grader_model: str):
    with _RAGAS_LLM_LOCK:
        if grader_model not in _RAGAS_LLM:
            _RAGAS_LLM[grader_model] = make_ragas_llm(grader_model)
        return _RAGAS_LLM[grader_model]


def _score_ragas_faithfulness(row: AuditRow, grader_model: str) -> tuple[float, str]:
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness

    metric = Faithfulness(llm=_ragas_llm(grader_model))
    sample = SingleTurnSample(
        user_input=row.question,
        response=row.raw_response,
        retrieved_contexts=[row.context],
    )
    score = asyncio.run(metric.single_turn_ascore(sample))
    return float(score), ""


def _score_ragas_correctness(row: AuditRow, grader_model: str) -> tuple[float, str]:
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import AnswerCorrectness

    metric = AnswerCorrectness(llm=_ragas_llm(grader_model), weights=[1.0, 0.0])
    sample = SingleTurnSample(
        user_input=row.question,
        response=row.raw_response,
        reference=row.gold_standard,
    )
    score = asyncio.run(metric.single_turn_ascore(sample))
    return float(score), ""


def _score_one(metric: str, row: AuditRow, grader_model: str) -> tuple[float, str]:
    if metric == "geval_strict":
        return _score_geval(row, _GEVAL_STRICT_STEPS, "compliance_correctness_strict", grader_model)
    if metric == "geval_lenient":
        return _score_geval(row, _GEVAL_LENIENT_STEPS, "conclusion_correctness_lenient", grader_model)
    if metric == "ragas_faithfulness":
        return _score_ragas_faithfulness(row, grader_model)
    if metric == "ragas_answer_correctness":
        return _score_ragas_correctness(row, grader_model)
    raise ValueError(metric)


def run_metrics(
    rows: list[AuditRow],
    out_path: Path,
    *,
    grader_model: str = DEFAULT_GRADER_MODEL,
    parallelism: int = 4,
    metrics: tuple[str, ...] = METRICS,
) -> int:
    """Score every (row, metric) pair not already in out_path. Returns #new."""
    done: set[tuple[str, str, str]] = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["metric"], r["examinee_id"], r["task_id"]))

    todo = [
        (metric, row)
        for metric in metrics
        for row in rows
        if (metric, row.examinee_id, row.task_id) not in done
    ]
    print(f"[metrics] {len(done)} already scored, {len(todo)} to score")
    if not todo:
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    n_written = 0
    t0 = time.time()

    def work(job: tuple[str, AuditRow]) -> None:
        nonlocal n_written
        metric, row = job
        try:
            score, reason = _score_one(metric, row, grader_model)
            err = None
        except Exception as e:  # noqa: BLE001 — log and continue; resumable
            score, reason, err = None, "", f"{type(e).__name__}: {e}"
        entry = {
            "metric": metric,
            "examinee_id": row.examinee_id,
            "task_id": row.task_id,
            "score": score,
            "reason": reason[:500],
            "error": err,
            "grader_model": grader_model,
        }
        with write_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
            n_written += 1
            if n_written % 50 == 0:
                rate = n_written / (time.time() - t0)
                eta = (len(todo) - n_written) / rate / 60 if rate > 0 else 0
                print(f"[metrics] {n_written}/{len(todo)}  ({rate:.1f}/s, ~{eta:.0f} min left)")

    with ThreadPoolExecutor(max_workers=parallelism) as ex:
        list(ex.map(work, todo))
    return n_written

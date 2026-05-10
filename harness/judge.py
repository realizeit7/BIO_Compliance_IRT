"""
harness/judge.py — LLM-as-judge for binary PASS/FAIL grading.

Ported from legacy evaluator.py with one architectural change: the judge now
returns a structured JudgeVerdict and never raises. Any failure (parse error,
empty response, API error) is mapped to FAIL with error metadata, per
CRITICAL ENGINEERING RULE 3.

The judge lives in /harness/ rather than /evaluator/ because it makes a
network call to an LLM. /evaluator/ is OFFLINE ONLY (psychometrics math on
logged jsonl data — no API calls).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from harness.groq_client import GroqClient

_JUDGE_SYSTEM_STRICT = (
    "You are an impartial regulatory compliance examiner. "
    "Your sole job is to decide whether a response correctly addresses "
    "a biopharma regulatory question. Focus on whether the core compliance "
    "determination is correct (compliant vs. non-compliant) and whether the "
    "response identifies the right regulatory obligation. Mark FAIL if the "
    "compliance conclusion is wrong, a key obligation is completely missed, "
    "no specific regulatory citation is provided, or the answer is vague to "
    "the point of being useless."
)

_JUDGE_TEMPLATE_STRICT = """\
You must evaluate whether the SOLVER RESPONSE correctly answers the compliance question
and satisfies the GOLD STANDARD requirement.

--- QUESTION ---
{question}

--- GOLD STANDARD (the correct, complete answer) ---
{gold_standard}

--- SOLVER RESPONSE ---
{response}

Evaluation criteria:
1. Does the response reach the correct compliance conclusion (compliant vs non-compliant, major vs minor deviation)?
2. Does it cite at least one specific regulation at the SECTION level (e.g., "21 CFR 11.10", "21 CFR 211.25", "ICH E6(R2) §4.8.2", "45 CFR 46.116")? Citing only a Part or Title (e.g., "21 CFR Part 11", "GMP regulations") WITHOUT a specific section number does NOT satisfy this criterion.
3. Is the substantive compliance reasoning consistent with the gold standard's key obligation?

Mark PASS only when ALL THREE criteria are met.
Mark FAIL otherwise.

Respond with EXACTLY this format (two lines, nothing else):
VERDICT: <PASS or FAIL>
REASON: <one sentence explaining the verdict>
"""

_JUDGE_SYSTEM_LENIENT = (
    "You are an impartial regulatory compliance examiner grading straightforward "
    "biopharma compliance questions. Focus solely on whether the response reaches "
    "the correct compliance conclusion with coherent reasoning. Specific section "
    "citations are NOT required. Mark FAIL only if the conclusion is wrong or the "
    "answer is too vague to convey any meaningful determination."
)

_JUDGE_TEMPLATE_LENIENT = """\
You must evaluate whether the SOLVER RESPONSE correctly answers the compliance question
and satisfies the GOLD STANDARD requirement.

--- QUESTION ---
{question}

--- GOLD STANDARD (the correct, complete answer) ---
{gold_standard}

--- SOLVER RESPONSE ---
{response}

Evaluation criteria:
1. Does the response reach the correct compliance conclusion?
2. Is the substantive reasoning consistent with the gold standard's key obligation?

Mark PASS when BOTH criteria are met. Specific regulatory citations are NOT required.
Mark FAIL only when the conclusion is wrong or the answer is too vague to be useful.

Respond with EXACTLY this format (two lines, nothing else):
VERDICT: <PASS or FAIL>
REASON: <one sentence explaining the verdict>
"""


@dataclass
class JudgeVerdict:
    """Structured judge output with full provenance."""

    passed: bool
    reason: str
    raw_text: str
    judge_model: str
    retry_count: int
    error_type: str | None = None


class Judge:
    """LLM-as-judge wrapper around GroqClient."""

    def __init__(
        self,
        client: GroqClient,
        *,
        model: str = "llama-3.3-70b-versatile",
        strict: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.strict = strict
        self._system = _JUDGE_SYSTEM_STRICT if strict else _JUDGE_SYSTEM_LENIENT
        self._template = _JUDGE_TEMPLATE_STRICT if strict else _JUDGE_TEMPLATE_LENIENT

    def grade(self, *, question: str, gold_standard: str, response: str) -> JudgeVerdict:
        if not response or len(response.strip()) < 10:
            return JudgeVerdict(
                passed=False,
                reason="Response was empty or too short to evaluate.",
                raw_text="",
                judge_model=self.model,
                retry_count=0,
                error_type="EmptyResponse",
            )

        prompt = self._template.format(
            question=question, gold_standard=gold_standard, response=response
        )
        result = self.client.generate(
            model=self.model,
            messages=[
                {"role": "system", "content": self._system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )

        if result.error_type is not None or not result.text:
            return JudgeVerdict(
                passed=False,
                reason=f"Judge harness error: {result.error_type}",
                raw_text=result.text,
                judge_model=self.model,
                retry_count=result.retry_count,
                error_type=result.error_type or "EmptyJudgeOutput",
            )

        passed, reason = self._parse_verdict(result.text)
        return JudgeVerdict(
            passed=passed,
            reason=reason,
            raw_text=result.text,
            judge_model=self.model,
            retry_count=result.retry_count,
            error_type=None,
        )

    @staticmethod
    def _parse_verdict(raw: str) -> tuple[bool, str]:
        verdict_match = re.search(r"VERDICT:\s*(PASS|FAIL)", raw, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
        if not verdict_match:
            upper = raw.upper()
            return ("PASS" in upper and "FAIL" not in upper), raw.strip()
        passed = verdict_match.group(1).upper() == "PASS"
        reason = reason_match.group(1).strip() if reason_match else raw.strip()
        return passed, reason

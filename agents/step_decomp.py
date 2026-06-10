"""
agents/step_decomp.py — Step-Decomposition Agent for Phase 3b.

Breaks the compliance analysis into four sequential Groq calls, each
building on the previous step's output. The final (step 4) answer is
graded by the Judge.

Steps
-----
1. Identify the regulated activity and parties.
2. Identify the specific CFR/ICH provision governing the activity.
3. Assess whether the activity meets or violates that provision.
4. State the final compliance determination with section-level citation.
"""

from __future__ import annotations

from agents.base import Agent, AgentResponse, Item
from agents.zero_shot import STRICTNESS_LEVELS
from harness.groq_client import GroqClient

_STEP_PROMPTS = [
    "Identify the regulated activity and the parties involved in this scenario.",
    "Identify the specific CFR provision or regulatory requirement that governs this activity.",
    "Assess whether the activity described meets or violates that requirement. Explain your reasoning.",
    (
        "State your final compliance determination. "
        "Include the specific section-level citation (e.g., '21 CFR 11.10(e)') "
        "and a clear compliant / non-compliant conclusion."
    ),
]


class StepDecompositionAgent(Agent):
    """
    Four sequential Groq calls forming a chain-of-thought scaffold.
    Each step is appended to the conversation before the next question.
    """

    def __init__(
        self,
        *,
        examinee_id: str,
        model: str,
        temperature: float,
        strictness: str,
        client: GroqClient,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> None:
        if strictness not in STRICTNESS_LEVELS:
            raise ValueError(
                f"Unknown strictness level: {strictness!r}. "
                f"Must be one of {sorted(STRICTNESS_LEVELS)}."
            )
        self.examinee_id = examinee_id
        self.model = model
        self.temperature = temperature
        self.strictness = strictness
        self.client = client
        self.max_tokens = max_tokens
        self.seed = seed
        self._system_prompt = STRICTNESS_LEVELS[strictness]

    def solve(self, item: Item) -> AgentResponse:
        context_block = f"{item.context}\n\nQuestion: {item.question}"

        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        # Seed the conversation with the scenario
        messages.append({"role": "user", "content": context_block})

        total_latency = 0.0
        total_retries = 0
        last_error_type = None
        last_error_message = None
        last_finish_reason = None
        last_raw_meta: dict = {}
        final_text = ""

        for i, step_prompt in enumerate(_STEP_PROMPTS):
            if i == 0:
                # First step: the scenario was already added; append step question
                messages.append({"role": "user", "content": step_prompt})
            else:
                messages.append({"role": "user", "content": step_prompt})

            result = self.client.generate(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
            )
            total_latency += result.latency_seconds
            total_retries += result.retry_count
            if result.error_type:
                last_error_type = result.error_type
                last_error_message = result.error_message
            last_finish_reason = result.finish_reason
            last_raw_meta = result.raw_meta or {}
            final_text = result.text

            messages.append({"role": "assistant", "content": result.text})

        return AgentResponse(
            text=final_text,
            retry_count=total_retries,
            error_type=last_error_type,
            error_message=last_error_message,
            latency_seconds=total_latency,
            finish_reason=last_finish_reason,
            extra={
                "strictness": self.strictness,
                "seed": self.seed,
                "raw_meta": last_raw_meta,
            },
        )

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "agent_type": "step_decomp",
            "strictness": self.strictness,
            "seed": self.seed,
        }

"""
agents/critic.py — Self-Critique Agent for Phase 3b.

Makes an initial zero-shot answer, then calls itself once per critique
round to review and correct before producing a final synthesis.

Default: 1 critique round = 3 total Groq calls per item.
"""

from __future__ import annotations

from agents.base import Agent, AgentResponse, Item
from agents.zero_shot import STRICTNESS_LEVELS, _USER_TEMPLATE
from harness.groq_client import GroqClient

_CRITIQUE_PROMPT = (
    "Review your previous answer for citation accuracy and compliance "
    "determination. Correct any errors, add any missing CFR/ICH citations, "
    "and confirm or revise your compliant / non-compliant determination."
)

_SYNTHESIS_PROMPT = (
    "Based on your analysis above, provide your final compliance determination "
    "with section-level CFR/ICH citations. Be concise and direct."
)


class CriticAgent(Agent):
    """
    Multi-call agent: initial answer → n_rounds critique loops → final synthesis.
    All calls share the same conversation context (multi-turn).
    """

    def __init__(
        self,
        *,
        examinee_id: str,
        model: str,
        temperature: float,
        strictness: str,
        client: GroqClient,
        n_rounds: int = 1,
        max_tokens: int = 1024,
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
        self.n_rounds = n_rounds
        self.max_tokens = max_tokens
        self.seed = seed
        self._system_prompt = STRICTNESS_LEVELS[strictness]

    def solve(self, item: Item) -> AgentResponse:
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})

        # Turn 1 — initial answer
        user_msg = _USER_TEMPLATE.format(context=item.context, question=item.question)
        messages.append({"role": "user", "content": user_msg})

        total_latency = 0.0
        total_retries = 0
        last_error_type = None
        last_error_message = None
        last_finish_reason = None
        last_raw_meta: dict = {}

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

        messages.append({"role": "assistant", "content": result.text})

        # Critique rounds
        for _ in range(self.n_rounds):
            messages.append({"role": "user", "content": _CRITIQUE_PROMPT})
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
            messages.append({"role": "assistant", "content": result.text})

        # Final synthesis
        messages.append({"role": "user", "content": _SYNTHESIS_PROMPT})
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

        return AgentResponse(
            text=result.text,
            retry_count=total_retries,
            error_type=last_error_type,
            error_message=last_error_message,
            latency_seconds=total_latency,
            finish_reason=last_finish_reason,
            extra={
                "strictness": self.strictness,
                "n_rounds": self.n_rounds,
                "seed": self.seed,
                "raw_meta": last_raw_meta,
            },
        )

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "agent_type": "critic",
            "strictness": self.strictness,
            "n_rounds": self.n_rounds,
            "seed": self.seed,
        }

"""
agents/zero_shot.py — Phase 1 naked baseline agent.

Single LLM call with a system prompt selected by `strictness` and the item's
context+question as the user message. No tools, no retries beyond the
harness layer, no chain-of-thought scaffolding.

Strictness levels
-----------------
* "none"    — empty system prompt; closest to a raw API call
* "neutral" — generic helpful-assistant prompt
* "strict"  — instructs the model to behave as a regulatory compliance expert
              and cite specific CFR/ICH sections at section level

These three levels are one of three grid axes (Models × Temperatures ×
Strictness) that produce the virtual examinee population required to
satisfy IRT sample-size requirements (Critical Engineering Rule 1).
"""

from __future__ import annotations

from agents.base import Agent, AgentResponse, Item
from harness.groq_client import GroqClient

STRICTNESS_LEVELS: dict[str, str] = {
    "none": "",
    "neutral": "You are a helpful assistant. Answer the user's question directly and concisely.",
    "strict": (
        "You are a senior regulatory compliance expert specialising in FDA "
        "biopharma regulations (21 CFR Parts 11, 50, 56, 211; ICH E6(R2) GCP; "
        "GMP). Answer the user's compliance question precisely. Always cite the "
        "specific regulation at the section level (e.g., '21 CFR 11.10(e)', "
        "'ICH E6(R2) §4.8.2'). Reach a clear compliant / non-compliant "
        "determination and state the key obligation."
    ),
}

_USER_TEMPLATE = """{context}

Question: {question}

Answer directly and concisely."""


class ZeroShotAgent(Agent):
    """
    Single-call agent. Naked baseline = (model, temperature, strictness)
    fully determine its behaviour.
    """

    def __init__(
        self,
        *,
        examinee_id: str,
        model: str,
        temperature: float,
        strictness: str,
        client: GroqClient,
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
        self.max_tokens = max_tokens
        self.seed = seed
        self._system_prompt = STRICTNESS_LEVELS[strictness]

    def solve(self, item: Item) -> AgentResponse:
        user_msg = _USER_TEMPLATE.format(context=item.context, question=item.question)

        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": user_msg})

        result = self.client.generate(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        return AgentResponse(
            text=result.text,
            retry_count=result.retry_count,
            error_type=result.error_type,
            error_message=result.error_message,
            latency_seconds=result.latency_seconds,
            finish_reason=result.finish_reason,
            extra={
                "strictness": self.strictness,
                "seed": self.seed,
                "raw_meta": result.raw_meta,
            },
        )

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "strictness": self.strictness,
            "seed": self.seed,
        }

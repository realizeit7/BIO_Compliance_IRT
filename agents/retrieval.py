"""
agents/retrieval.py — Retrieval-Augmented Agent for Phase 3b.

Fetches the top-k CFR sections most relevant to the item before calling
the LLM, prepending them as a "Relevant Regulations" block in the system
prompt. Everything else mirrors ZeroShotAgent so Δθ comparisons are clean.
"""

from __future__ import annotations

from agents.base import Agent, AgentResponse, Item
from agents.zero_shot import STRICTNESS_LEVELS, _USER_TEMPLATE
from harness.cfr_store import CFRStore
from harness.groq_client import GroqClient

_RAG_SYSTEM_PREFIX = """\
Relevant Regulations:
{sections}

---
"""

_SECTION_TEMPLATE = "[{section_id}] {title}\n{text}"


class RetrievalAgent(Agent):
    """
    Single LLM call, same as ZeroShotAgent, but the system prompt is
    augmented with top-k CFR sections retrieved by cosine similarity
    over the pre-built sentence-transformer index.
    """

    def __init__(
        self,
        *,
        examinee_id: str,
        model: str,
        temperature: float,
        strictness: str,
        client: GroqClient,
        cfr_store: CFRStore,
        k: int = 3,
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
        self.cfr_store = cfr_store
        self.k = k
        self.max_tokens = max_tokens
        self.seed = seed
        self._base_system = STRICTNESS_LEVELS[strictness]

    def solve(self, item: Item) -> AgentResponse:
        query = f"{item.context}\n{item.question}"
        sections = self.cfr_store.retrieve(query, k=self.k)

        sections_text = "\n\n".join(
            _SECTION_TEMPLATE.format(
                section_id=s.section_id,
                title=s.title,
                text=s.text[:600],  # cap to avoid token bloat
            )
            for s in sections
        )
        rag_prefix = _RAG_SYSTEM_PREFIX.format(sections=sections_text)
        system_prompt = rag_prefix + self._base_system

        user_msg = _USER_TEMPLATE.format(context=item.context, question=item.question)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

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
                "retrieved_sections": [s.section_id for s in sections],
                "raw_meta": result.raw_meta,
            },
        )

    def describe(self) -> dict[str, object]:
        return {
            **super().describe(),
            "agent_type": "retrieval",
            "strictness": self.strictness,
            "k": self.k,
            "seed": self.seed,
        }

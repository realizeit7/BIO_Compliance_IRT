"""
agents/base.py — Agent ABC and shared dataclasses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Item:
    """A compliance item presented to the agent."""

    task_id: str
    domain: str
    context: str
    question: str
    gold_standard: str
    source_type: str  # "synthetic" | "real"
    source_ref: str | None = None


@dataclass
class AgentResponse:
    """Output of one Agent.solve() call, including provenance."""

    text: str
    retry_count: int
    error_type: str | None
    error_message: str | None
    latency_seconds: float
    finish_reason: str | None
    extra: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """
    Abstract Agent. Subclasses implement an orchestration strategy.

    Identity (`examinee_id`) is set by the Arena at construction time so
    that all calls from a given grid configuration share a stable ID
    used for IRT calibration.
    """

    examinee_id: str
    model: str
    temperature: float

    @abstractmethod
    def solve(self, item: Item) -> AgentResponse:
        """Process one item and return a response."""
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        """Return JSON-serialisable metadata for log entries."""
        return {
            "examinee_id": self.examinee_id,
            "agent_class": type(self).__name__,
            "model": self.model,
            "temperature": self.temperature,
        }

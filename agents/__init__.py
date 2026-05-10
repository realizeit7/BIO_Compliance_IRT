"""
Agents layer — orchestration and control flow.

An Agent is anything that consumes an Item and produces a final response
string. Agent strategies range from naked single-call (Phase 1 baseline)
to multi-step reasoning loops, tool use, and PydanticAI workflows
(Phase 3+).

Agents call the /harness/ layer for all LLM I/O. They never touch the
network directly.
"""

from agents.base import Agent, AgentResponse, Item
from agents.zero_shot import ZeroShotAgent, STRICTNESS_LEVELS

__all__ = [
    "Agent",
    "AgentResponse",
    "Item",
    "ZeroShotAgent",
    "STRICTNESS_LEVELS",
]

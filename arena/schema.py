"""
arena/schema.py — Pydantic log entry schema.

One ArenaLogEntry per (examinee × item) attempt, written as one line of
jsonl. The schema is the public contract between the live Arena and the
offline /evaluator/. Adding fields is fine; renaming or removing existing
fields is a breaking change.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ArenaLogEntry(BaseModel):
    """One graded attempt: (run_id, examinee_id, task_id) is the natural key."""

    model_config = ConfigDict(extra="forbid")

    # Provenance
    run_id: str
    examinee_id: str
    task_id: str
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Examinee (grid axes)
    agent_class: str
    model: str
    temperature: float
    strictness: str | None = None
    seed: int | None = None

    # Item
    domain: str
    source_type: str  # "synthetic" | "real"

    # Generation
    prompt_messages: list[dict[str, str]] | None = None
    raw_response: str
    retry_count: int = 0
    finish_reason: str | None = None
    latency_seconds: float = 0.0
    solver_error_type: str | None = None
    solver_error_message: str | None = None

    # Judge
    judge_model: str
    judge_strict: bool
    judge_raw: str = ""
    judge_reason: str = ""
    judge_error_type: str | None = None

    # Final binary outcome — what /evaluator/ reads
    parsed_result: int  # 0 = FAIL, 1 = PASS

    # Free-form bag for future extensions (kept opaque to the schema)
    extra: dict[str, Any] = Field(default_factory=dict)

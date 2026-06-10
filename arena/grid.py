"""
arena/grid.py — Virtual examinee grid generator.

Produces N = |models| × |temperatures| × |strictness_levels| × |agent_types|
ExamineeConfig entries. Each config gets a deterministic examinee_id derived from
its grid coordinates so that logs from independent runs collate cleanly.

Phase 1 baseline: agent_types defaults to ["zero_shot"] and the ID format is
backward-compatible (agent_type is omitted from the hash for "zero_shot").
Phase 3b: pass agent_types=["zero_shot","retrieval","critic","step_decomp"].
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ExamineeConfig:
    """One point on the (model × temperature × strictness × agent_type) grid."""

    examinee_id: str
    model: str
    temperature: float
    strictness: str
    agent_type: str = "zero_shot"
    seed: int | None = None

    def to_dict(self) -> dict:
        return {
            "examinee_id": self.examinee_id,
            "model": self.model,
            "temperature": self.temperature,
            "strictness": self.strictness,
            "agent_type": self.agent_type,
            "seed": self.seed,
        }


def _make_examinee_id(
    model: str,
    temperature: float,
    strictness: str,
    agent_type: str = "zero_shot",
) -> str:
    """Stable short ID derived from grid coordinates.

    For agent_type='zero_shot' the hash matches the Phase 1 format exactly
    (backward-compatible). Other agent types include the type in the hash.
    """
    if agent_type == "zero_shot":
        raw = f"{model}|t={temperature:.2f}|s={strictness}"
    else:
        raw = f"{model}|t={temperature:.2f}|s={strictness}|a={agent_type}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    safe_model = model.replace("/", "_").replace("-", "_")
    return f"{safe_model}_t{temperature:.2f}_s{strictness}_{h}"


def build_grid(
    *,
    models: list[str],
    temperatures: list[float],
    strictness_levels: list[str],
    agent_types: list[str] | None = None,
    seed: int | None = None,
) -> list[ExamineeConfig]:
    """Return the full Cartesian-product grid of examinee configs."""
    if agent_types is None:
        agent_types = ["zero_shot"]
    grid: list[ExamineeConfig] = []
    for model in models:
        for temp in temperatures:
            for strict in strictness_levels:
                for atype in agent_types:
                    grid.append(
                        ExamineeConfig(
                            examinee_id=_make_examinee_id(model, temp, strict, atype),
                            model=model,
                            temperature=float(temp),
                            strictness=strict,
                            agent_type=atype,
                            seed=seed,
                        )
                    )
    return grid


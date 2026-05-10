"""
arena/grid.py — Virtual examinee grid generator.

Produces N = |models| × |temperatures| × |strictness_levels| ZeroShotAgent
configurations. Each config gets a deterministic examinee_id derived from
its grid coordinates so that logs from independent runs collate cleanly.

This satisfies Critical Engineering Rule 1 (N ≥ 200 via grid expansion)
without requiring 200+ distinct base models.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ExamineeConfig:
    """One point on the (model × temperature × strictness) grid."""

    examinee_id: str
    model: str
    temperature: float
    strictness: str
    seed: int | None = None

    def to_dict(self) -> dict:
        return {
            "examinee_id": self.examinee_id,
            "model": self.model,
            "temperature": self.temperature,
            "strictness": self.strictness,
            "seed": self.seed,
        }


def _make_examinee_id(model: str, temperature: float, strictness: str) -> str:
    """Stable short ID derived from grid coordinates."""
    raw = f"{model}|t={temperature:.2f}|s={strictness}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    safe_model = model.replace("/", "_").replace("-", "_")
    return f"{safe_model}_t{temperature:.2f}_s{strictness}_{h}"


def build_grid(
    *,
    models: list[str],
    temperatures: list[float],
    strictness_levels: list[str],
    seed: int | None = None,
) -> list[ExamineeConfig]:
    """Return the full Cartesian-product grid of examinee configs."""
    grid: list[ExamineeConfig] = []
    for model in models:
        for temp in temperatures:
            for strict in strictness_levels:
                grid.append(
                    ExamineeConfig(
                        examinee_id=_make_examinee_id(model, temp, strict),
                        model=model,
                        temperature=float(temp),
                        strictness=strict,
                        seed=seed,
                    )
                )
    return grid

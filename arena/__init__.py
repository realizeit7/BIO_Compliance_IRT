"""
Arena layer — the dynamic testing loop.

Reads items (read-only) from compliance_bank.db, generates the virtual
examinee grid, drives each (examinee × item) pair through Agent → Judge,
and writes one structured jsonl line per attempt for the offline
/evaluator/ to consume.

The Arena is the only layer that touches both the legacy DB and the
runtime LLM Harness. It writes append-only jsonl logs with full
provenance for full reproducibility (Critical Engineering Rule 2).
"""

from arena.schema import ArenaLogEntry
from arena.loader import ItemLoader
from arena.grid import ExamineeConfig, build_grid
from arena.runner import ArenaRunner

__all__ = [
    "ArenaLogEntry",
    "ItemLoader",
    "ExamineeConfig",
    "build_grid",
    "ArenaRunner",
]

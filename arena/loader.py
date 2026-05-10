"""
arena/loader.py — Read-only DB loader.

Opens compliance_bank.db in URI mode=ro to guarantee the legacy DB is
treated as Read-Only context (per Critical Engineering Rule 4). Returns
a list of agents.Item dicts, optionally stratified-sampled.

Items are sourced from the `tasks` table only — calibration history in
`calibration_results` is irrelevant to the new Arena pipeline because
we are re-validating every item from scratch.
"""

from __future__ import annotations

import random
import sqlite3
from collections import defaultdict
from pathlib import Path

from agents.base import Item


class ItemLoader:
    """Read-only loader for compliance_bank.db tasks."""

    def __init__(self, db_path: str | Path = "compliance_bank.db") -> None:
        self.db_path = str(Path(db_path).resolve())
        # mode=ro enforces read-only at the SQLite layer; any write would error.
        uri = f"file:{self.db_path}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def fetch_all(
        self,
        *,
        source_types: tuple[str, ...] = ("synthetic", "real"),
    ) -> list[Item]:
        placeholders = ",".join("?" * len(source_types))
        rows = self._conn.execute(
            f"""
            SELECT task_id, domain, context, question, gold_standard,
                   source_type, source_ref
            FROM tasks
            WHERE source_type IN ({placeholders})
            ORDER BY task_id
            """,
            source_types,
        ).fetchall()
        return [
            Item(
                task_id=r["task_id"],
                domain=r["domain"] or "unknown",
                context=r["context"],
                question=r["question"],
                gold_standard=r["gold_standard"],
                source_type=r["source_type"] or "synthetic",
                source_ref=r["source_ref"],
            )
            for r in rows
        ]

    def stratified_sample(
        self,
        *,
        sample_size: int,
        seed: int = 42,
        source_types: tuple[str, ...] = ("synthetic", "real"),
        stratify_by: tuple[str, ...] = ("domain", "source_type"),
    ) -> list[Item]:
        """
        Return a stratified sample of size approximately `sample_size`.

        Allocates per-stratum quota proportional to stratum size; rounds
        up to guarantee at least one item per stratum that has any items.
        Random selection within each stratum uses the given seed.
        """
        items = self.fetch_all(source_types=source_types)
        if sample_size >= len(items):
            return items

        rng = random.Random(seed)

        def key(it: Item) -> tuple:
            return tuple(getattr(it, axis) for axis in stratify_by)

        buckets: dict[tuple, list[Item]] = defaultdict(list)
        for it in items:
            buckets[key(it)].append(it)

        total = len(items)
        sampled: list[Item] = []
        for k, bucket in buckets.items():
            quota = max(1, round(sample_size * len(bucket) / total))
            quota = min(quota, len(bucket))
            sampled.extend(rng.sample(bucket, quota))

        # If proportional rounding overshot, trim deterministically
        if len(sampled) > sample_size:
            rng.shuffle(sampled)
            sampled = sampled[:sample_size]

        return sampled

    def close(self) -> None:
        self._conn.close()

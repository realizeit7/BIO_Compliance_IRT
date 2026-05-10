"""
evaluator/response_matrix.py — Build the binary response matrix from jsonl logs.

Pure-Python utility: reads append-only jsonl files written by /arena/,
deduplicates on (examinee_id, task_id) keeping the LAST entry, and
returns a ResponseMatrix in girth-friendly [items × examinees] layout.

NEVER calls an LLM. NEVER imports from /harness/ or /agents/.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


@dataclass
class ResponseMatrix:
    """Binary response matrix in [items × examinees] orientation (girth convention)."""

    matrix: np.ndarray  # shape (n_items, n_examinees), dtype int8 ∈ {0, 1}
    item_ids: list[str]
    examinee_ids: list[str]

    @property
    def n_items(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_examinees(self) -> int:
        return self.matrix.shape[1]

    def examinee_index(self, examinee_id: str) -> int:
        return self.examinee_ids.index(examinee_id)

    def item_index(self, task_id: str) -> int:
        return self.item_ids.index(task_id)

    def baseline_responses(self, examinee_id: str) -> np.ndarray:
        """Return this examinee's response vector across all items, shape (n_items,)."""
        return self.matrix[:, self.examinee_index(examinee_id)]


def load_jsonl(paths: str | Path | Iterable[str | Path]) -> Iterator[dict]:
    """Yield decoded jsonl records from one or many paths."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for p in paths:
        p = Path(p)
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def build_response_matrix(
    records: Iterable[dict],
    *,
    require_judge_ok: bool = False,
) -> ResponseMatrix:
    """
    Convert jsonl records into a (items × examinees) binary matrix.

    Parameters
    ----------
    records         : iterable of decoded log dicts (from load_jsonl).
    require_judge_ok: when True, drop entries where the judge itself errored
                      (judge_error_type is not null). Default False — judge
                      errors map to FAIL per Rule 3, which is what we want
                      for the live IRT calibration. Set True only for
                      diagnostic analyses.

    Later entries for the same (examinee, item) overwrite earlier ones, so
    a resumed run yields the most recent attempt's verdict.
    """
    by_pair: dict[tuple[str, str], int] = {}
    item_set: set[str] = set()
    examinee_set: set[str] = set()

    for rec in records:
        eid = rec["examinee_id"]
        tid = rec["task_id"]
        if require_judge_ok and rec.get("judge_error_type"):
            continue
        result = int(rec["parsed_result"])
        by_pair[(eid, tid)] = result
        item_set.add(tid)
        examinee_set.add(eid)

    item_ids = sorted(item_set)
    examinee_ids = sorted(examinee_set)
    item_idx = {t: i for i, t in enumerate(item_ids)}
    examinee_idx = {e: j for j, e in enumerate(examinee_ids)}

    # girth treats missing values as NaN; we use -1 as a sentinel mapped after.
    mat = np.full((len(item_ids), len(examinee_ids)), -1, dtype=np.int8)
    for (eid, tid), val in by_pair.items():
        mat[item_idx[tid], examinee_idx[eid]] = val

    return ResponseMatrix(matrix=mat, item_ids=item_ids, examinee_ids=examinee_ids)

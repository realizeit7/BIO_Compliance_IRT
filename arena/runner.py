"""
arena/runner.py — Resumable Arena loop.

For each (examinee × item) pair: builds a ZeroShotAgent, calls solve(),
hands the response to the Judge, writes one ArenaLogEntry to the run's
jsonl file. Skips pairs already present in the log so interrupted runs
can resume cleanly.

Per Critical Engineering Rule 3, NO exception escapes here: any solver
or judge failure is mapped to parsed_result=0 and logged with the
error_type / error_message metadata.
"""

from __future__ import annotations

import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

from tqdm import tqdm

from agents.base import Agent, Item
from agents.zero_shot import ZeroShotAgent
from arena.grid import ExamineeConfig
from arena.schema import ArenaLogEntry
from harness.groq_client import GroqClient
from harness.judge import Judge


class ArenaRunner:
    """
    Drives the full grid × items loop with resume support and parallel
    in-flight requests.

    Parameters
    ----------
    run_id        : Identifier for this run; jsonl path is
                    {log_dir}/{run_id}/responses.jsonl
    log_dir       : Root directory for arena run logs.
    client        : Shared GroqClient (one connection pool, reused across agents).
    judge         : Shared Judge (strict mode for hard items, lenient for easy).
    judge_lenient : Optional second Judge instance used when item.domain starts
                    with 'easy_'. If None, the same `judge` is used everywhere.
    parallelism   : Max concurrent in-flight (solver + judge) calls.
    """

    def __init__(
        self,
        *,
        run_id: str,
        log_dir: str | Path,
        client: GroqClient,
        judge: Judge,
        judge_lenient: Judge | None = None,
        parallelism: int = 8,
        agent_factory: Callable[[ExamineeConfig], Agent] | None = None,
    ) -> None:
        self.run_id = run_id
        self.client = client
        self.judge_strict = judge
        self.judge_lenient = judge_lenient or judge
        self.parallelism = parallelism
        self._agent_factory = agent_factory

        self.run_dir = Path(log_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "responses.jsonl"
        self._write_lock = threading.Lock()

    def already_done(self) -> set[tuple[str, str]]:
        """Return set of (examinee_id, task_id) pairs already logged."""
        done: set[tuple[str, str]] = set()
        if not self.log_path.exists():
            return done
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    done.add((obj["examinee_id"], obj["task_id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
        return done

    def run(
        self,
        *,
        examinees: list[ExamineeConfig],
        items: list[Item],
    ) -> int:
        """Execute the full grid × items product. Returns # new entries written."""
        done = self.already_done()
        pairs: list[tuple[ExamineeConfig, Item]] = [
            (e, it)
            for e in examinees
            for it in items
            if (e.examinee_id, it.task_id) not in done
        ]
        if not pairs:
            return 0

        total_planned = len(examinees) * len(items)
        print(
            f"[arena] run_id={self.run_id} pairs_to_run={len(pairs)} "
            f"already_done={len(done)} total={total_planned}"
        )

        n_written = 0
        with ThreadPoolExecutor(max_workers=self.parallelism) as pool:
            futures = [pool.submit(self._run_pair, e, it) for e, it in pairs]
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="arena", unit="pair"
            ):
                entry = fut.result()
                if entry is not None:
                    self._append(entry)
                    n_written += 1
        return n_written

    def _run_pair(self, examinee: ExamineeConfig, item: Item) -> ArenaLogEntry | None:
        """Run one (examinee, item) pair. Always returns an entry — never raises."""
        try:
            agent = self._build_agent(examinee)
            response = agent.solve(item)

            judge = self._select_judge(item)
            verdict = judge.grade(
                question=item.question,
                gold_standard=item.gold_standard,
                response=response.text,
            )

            entry = ArenaLogEntry(
                run_id=self.run_id,
                examinee_id=examinee.examinee_id,
                task_id=item.task_id,
                agent_class=type(agent).__name__,
                agent_type=examinee.agent_type,
                model=examinee.model,
                temperature=examinee.temperature,
                strictness=examinee.strictness,
                seed=examinee.seed,
                domain=item.domain,
                source_type=item.source_type,
                raw_response=response.text,
                retry_count=response.retry_count,
                finish_reason=response.finish_reason,
                latency_seconds=response.latency_seconds,
                solver_error_type=response.error_type,
                solver_error_message=response.error_message,
                judge_model=judge.model,
                judge_strict=judge.strict,
                judge_raw=verdict.raw_text,
                judge_reason=verdict.reason,
                judge_error_type=verdict.error_type,
                parsed_result=1 if verdict.passed else 0,
            )
            return entry
        except Exception as exc:  # noqa: BLE001 — Rule 3
            tb = traceback.format_exc()
            return ArenaLogEntry(
                run_id=self.run_id,
                examinee_id=examinee.examinee_id,
                task_id=item.task_id,
                agent_class=examinee.agent_type,
                agent_type=examinee.agent_type,
                model=examinee.model,
                temperature=examinee.temperature,
                strictness=examinee.strictness,
                seed=examinee.seed,
                domain=item.domain,
                source_type=item.source_type,
                raw_response="",
                retry_count=0,
                finish_reason=None,
                latency_seconds=0.0,
                solver_error_type=type(exc).__name__,
                solver_error_message=str(exc),
                judge_model=self.judge_strict.model,
                judge_strict=self.judge_strict.strict,
                judge_raw="",
                judge_reason=f"Runner-level exception: {exc}",
                judge_error_type="RunnerException",
                parsed_result=0,
                extra={"traceback": tb},
            )

    def _build_agent(self, examinee: ExamineeConfig) -> Agent:
        if self._agent_factory is not None:
            return self._agent_factory(examinee)
        return ZeroShotAgent(
            examinee_id=examinee.examinee_id,
            model=examinee.model,
            temperature=examinee.temperature,
            strictness=examinee.strictness,
            client=self.client,
            seed=examinee.seed,
        )

    def _select_judge(self, item: Item) -> Judge:
        return self.judge_lenient if item.domain.startswith("easy_") else self.judge_strict

    def _append(self, entry: ArenaLogEntry) -> None:
        line = entry.model_dump_json() + "\n"
        with self._write_lock:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line)


def iter_log(path: str | Path) -> Iterable[dict]:
    """Yield parsed jsonl entries from a run's responses.jsonl."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

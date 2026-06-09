"""
harness/groq_client.py — Raw Groq API wrapper.

Wraps openai.OpenAI pointed at https://api.groq.com/openai/v1 and provides:
  * generate(...) → GenerationResult with raw text + retry count + error metadata
  * Exponential backoff on retryable errors
  * Strict timeout handling
  * Exception → HarnessError mapping

This is the ONLY module in the project permitted to instantiate openai.OpenAI
for the Groq endpoint. Agents and Arena must use GroqClient.
"""

from __future__ import annotations

import os
import re
import time
import random
from dataclasses import dataclass, field
from typing import Any

import openai

from harness.errors import (
    HarnessAPIError,
    HarnessRateLimitError,
    HarnessTimeoutError,
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Two flavours of reasoning model on Groq:
#   1) Inline CoT (e.g. deepseek-r1-distill-llama-70b): emits <think>...</think> inside
#      message.content followed by the answer. We strip the block so callers see a
#      clean final answer.
#   2) Separate-channel CoT (e.g. openai/gpt-oss-*): emits chain-of-thought into
#      message.reasoning and the answer into message.content. With Groq's default
#      reasoning_effort="medium" the CoT regularly consumes the full max_tokens budget,
#      leaving message.content empty and finish_reason="length". We mitigate at two
#      layers: (a) auto-set reasoning_effort="low" for these models, (b) if content is
#      still empty but reasoning is populated, surface the reasoning text so the judge
#      has something to grade rather than a forced Fail(0).
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SEPARATE_REASONING_PREFIXES = ("openai/gpt-oss",)


def _strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks; trim whitespace."""
    return _THINK_BLOCK_RE.sub("", text).strip()


def _uses_separate_reasoning_channel(model: str) -> bool:
    return any(model.startswith(p) for p in _SEPARATE_REASONING_PREFIXES)


@dataclass
class GenerationResult:
    """One LLM generation, including provenance for reproducibility."""

    text: str
    model: str
    temperature: float
    seed: int | None
    retry_count: int
    latency_seconds: float
    finish_reason: str | None
    error_type: str | None = None
    error_message: str | None = None
    raw_meta: dict[str, Any] = field(default_factory=dict)


class GroqClient:
    """
    Thin wrapper over the Groq /chat/completions endpoint.

    Parameters
    ----------
    api_key      : Groq API key. Defaults to env GROQ_API_KEY.
    timeout      : Per-request timeout in seconds.
    max_retries  : Retries on retryable errors (rate limit, 5xx, timeout).
    backoff_base : Initial backoff in seconds; doubles each retry, plus jitter.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_base: float = 1.5,
    ) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Set it via: export GROQ_API_KEY=<your_key>"
            )
        self._client = openai.OpenAI(
            api_key=key,
            base_url=GROQ_BASE_URL,
            timeout=timeout,
        )
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> GenerationResult:
        """
        Run a chat completion and return a structured result.

        Never raises HarnessError — instead, packs the failure into the
        returned GenerationResult with text="" and error_type set. The
        Arena treats text=="" as Fail(0) at scoring time.
        """
        retry = 0
        last_error_type: str | None = None
        last_error_message: str | None = None
        start = time.perf_counter()

        while retry <= self.max_retries:
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if seed is not None:
                    kwargs["seed"] = seed
                if _uses_separate_reasoning_channel(model):
                    kwargs["reasoning_effort"] = "low"

                msg = self._client.chat.completions.create(**kwargs)
                choice = msg.choices[0]
                content = (choice.message.content or "").strip()
                reasoning = (getattr(choice.message, "reasoning", None) or "").strip()
                text = _strip_reasoning(content)
                used_reasoning_fallback = False
                if not text and reasoning:
                    text = _strip_reasoning(reasoning)
                    used_reasoning_fallback = True
                latency = time.perf_counter() - start
                return GenerationResult(
                    text=text,
                    model=model,
                    temperature=temperature,
                    seed=seed,
                    retry_count=retry,
                    latency_seconds=latency,
                    finish_reason=choice.finish_reason,
                    error_type=None,
                    error_message=None,
                    raw_meta={
                        "id": getattr(msg, "id", None),
                        "usage": getattr(msg, "usage", None).__dict__
                        if getattr(msg, "usage", None) is not None
                        else None,
                        "had_reasoning_block": content != text and not used_reasoning_fallback,
                        "used_reasoning_fallback": used_reasoning_fallback,
                    },
                )

            except openai.RateLimitError as exc:
                last_error_type = "RateLimitError"
                last_error_message = str(exc)
                if retry >= self.max_retries:
                    break
                self._sleep_backoff(retry)
                retry += 1

            except openai.APITimeoutError as exc:
                last_error_type = "TimeoutError"
                last_error_message = str(exc)
                if retry >= self.max_retries:
                    break
                self._sleep_backoff(retry)
                retry += 1

            except openai.APIStatusError as exc:
                # 5xx is retryable; 4xx (other than 429) is not
                status = getattr(exc, "status_code", None)
                last_error_type = f"APIStatusError({status})"
                last_error_message = str(exc)
                if status is not None and 500 <= status < 600 and retry < self.max_retries:
                    self._sleep_backoff(retry)
                    retry += 1
                    continue
                break

            except openai.APIConnectionError as exc:
                last_error_type = "APIConnectionError"
                last_error_message = str(exc)
                if retry >= self.max_retries:
                    break
                self._sleep_backoff(retry)
                retry += 1

            except Exception as exc:  # noqa: BLE001 — catch-all is intentional per Rule 3
                last_error_type = type(exc).__name__
                last_error_message = str(exc)
                break

        latency = time.perf_counter() - start
        return GenerationResult(
            text="",
            model=model,
            temperature=temperature,
            seed=seed,
            retry_count=retry,
            latency_seconds=latency,
            finish_reason=None,
            error_type=last_error_type or "UnknownError",
            error_message=last_error_message,
        )

    def _sleep_backoff(self, retry: int) -> None:
        delay = self.backoff_base * (2**retry) + random.uniform(0.0, 0.5)
        time.sleep(delay)

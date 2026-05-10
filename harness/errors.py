"""
harness/errors.py — Structured exception hierarchy for the Harness layer.

Every external failure mode (timeout, rate limit, API 5xx, malformed response,
model refusal) maps to a HarnessError subclass. The Arena converts any
HarnessError into a Fail(0) for the affected (examinee, item) pair, logs the
full trace to the run's jsonl, and continues — per CRITICAL ENGINEERING RULE 3.
"""

from __future__ import annotations


class HarnessError(Exception):
    """Base class for all Harness-layer failures. Always recoverable to Fail(0)."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class HarnessTimeoutError(HarnessError):
    """Request exceeded the configured timeout."""

    def __init__(self, message: str = "Request timed out") -> None:
        super().__init__(message, retryable=True)


class HarnessRateLimitError(HarnessError):
    """Provider returned 429 / quota exceeded."""

    def __init__(self, message: str = "Rate limit hit") -> None:
        super().__init__(message, retryable=True)


class HarnessAPIError(HarnessError):
    """Provider returned a 5xx or other recoverable HTTP error."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class HarnessParseError(HarnessError):
    """Model output could not be parsed into the expected structure."""

    def __init__(self, message: str = "Could not parse model output") -> None:
        super().__init__(message, retryable=False)


class HarnessRefusalError(HarnessError):
    """Model refused to answer (safety filter, policy block)."""

    def __init__(self, message: str = "Model refused to respond") -> None:
        super().__init__(message, retryable=False)

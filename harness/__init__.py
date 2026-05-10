"""
Harness layer — physical infrastructure between the Arena and external LLM APIs.

Responsibilities
----------------
* Wrap raw API calls (Groq via openai SDK).
* Enforce timeouts, retries, and structured error handling.
* Never crash the Arena: every external failure mode is mapped to a HarnessError
  subclass that the Arena converts to Fail(0) with a logged trace.

The Harness is the ONLY layer permitted to perform network I/O for LLM calls.
The /evaluator/ layer must never import from /harness/.
"""

from harness.errors import (
    HarnessError,
    HarnessTimeoutError,
    HarnessRateLimitError,
    HarnessAPIError,
    HarnessParseError,
    HarnessRefusalError,
)
from harness.groq_client import GroqClient, GenerationResult
from harness.judge import Judge, JudgeVerdict

__all__ = [
    "HarnessError",
    "HarnessTimeoutError",
    "HarnessRateLimitError",
    "HarnessAPIError",
    "HarnessParseError",
    "HarnessRefusalError",
    "GroqClient",
    "GenerationResult",
    "Judge",
    "JudgeVerdict",
]

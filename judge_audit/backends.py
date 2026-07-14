"""LLM backends for DeepEval and RAGAS pointed at Groq's OpenAI-compatible API.

No OpenAI key exists in this project; both tools run on the same Groq-hosted
model family as the project's own judge. That is a deliberate audit choice:
holding grader capability constant isolates the *framework* difference
(rubric, aggregation, score scale) — see METHODOLOGY.md.
"""

from __future__ import annotations

import os

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import AsyncOpenAI, OpenAI

from harness.groq_client import GROQ_BASE_URL

DEFAULT_GRADER_MODEL = "llama-3.3-70b-versatile"


def _api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    return key


class GroqDeepEvalLLM(DeepEvalBaseLLM):
    """Custom DeepEval model: Groq chat completions with JSON-schema parsing.

    GEval's custom-model path calls generate(prompt, schema=PydanticModel);
    we request JSON output and validate against the schema.
    """

    def __init__(self, model: str = DEFAULT_GRADER_MODEL, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._client = OpenAI(api_key=_api_key(), base_url=GROQ_BASE_URL)
        self._aclient = AsyncOpenAI(api_key=_api_key(), base_url=GROQ_BASE_URL)

    def load_model(self):
        return self._client

    def _completion_kwargs(self, prompt: str, schema) -> dict:
        kwargs: dict = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    @staticmethod
    def _parse(text: str, schema):
        if schema is None:
            return text
        # Strip code fences the model sometimes adds despite JSON mode
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return schema.model_validate_json(cleaned.strip())

    def generate(self, prompt: str, schema=None):
        resp = self._client.chat.completions.create(**self._completion_kwargs(prompt, schema))
        return self._parse(resp.choices[0].message.content or "", schema)

    async def a_generate(self, prompt: str, schema=None):
        resp = await self._aclient.chat.completions.create(**self._completion_kwargs(prompt, schema))
        return self._parse(resp.choices[0].message.content or "", schema)

    def get_model_name(self) -> str:
        return f"groq/{self.model}"


def make_ragas_llm(model: str = DEFAULT_GRADER_MODEL):
    """RAGAS LLM wrapper: langchain ChatOpenAI pointed at Groq."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(
        ChatOpenAI(
            model=model,
            temperature=0.0,
            api_key=_api_key(),
            base_url=GROQ_BASE_URL,
            max_retries=3,
            timeout=60,
        )
    )

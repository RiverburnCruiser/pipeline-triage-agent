"""The language model behind the agent, behind a two-method interface.

`ClaudeTriageModel` calls Claude through the official Anthropic SDK, and
langchain_llm.py wraps Gemini and Groq, which have free tiers. TRIAGE_PROVIDER picks
one. Tests use a scripted model with the same interface, so the graph, the MCP wiring
and the approval step are tested without an API key or network access.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Protocol, cast

from pipeline_triage.diagnosis import Diagnosis

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from anthropic.types.beta import BetaMessageParam, BetaOutputConfigParam, BetaToolUnionParam

Message = dict[str, Any]
"""One Messages API message as plain JSON, so LangGraph can checkpoint it."""

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class TriageModel(Protocol):
    async def next_turn(
        self, system: str, messages: list[Message], tools: list[dict[str, Any]]
    ) -> Message:
        """Return the assistant's next message: tool calls, or a final text answer."""
        ...

    async def diagnose(self, system: str, messages: list[Message]) -> Diagnosis:
        """Turn the investigation so far into a structured diagnosis."""
        ...


class ModelRefusedError(RuntimeError):
    pass


class ClaudeTriageModel:
    """Claude via the Messages API.

    Reads ANTHROPIC_API_KEY (or another credential the SDK finds). TRIAGE_MODEL and
    TRIAGE_EFFORT override the model and the effort level.
    """

    def __init__(
        self,
        model: str | None = None,
        effort: str | None = None,
        client: AsyncAnthropic | None = None,
    ) -> None:
        from anthropic import AsyncAnthropic

        self._client = client or AsyncAnthropic()
        self._model = model or os.environ.get("TRIAGE_MODEL", DEFAULT_MODEL)
        self._output_config = cast(
            "BetaOutputConfigParam",
            {"effort": effort or os.environ.get("TRIAGE_EFFORT", DEFAULT_EFFORT)},
        )

    async def next_turn(
        self, system: str, messages: list[Message], tools: list[dict[str, Any]]
    ) -> Message:
        response = await self._client.beta.messages.create(
            model=self._model,
            max_tokens=16000,
            system=system,
            messages=cast("list[BetaMessageParam]", messages),
            tools=cast("list[BetaToolUnionParam]", tools),
            output_config=self._output_config,
            # If a safety classifier declines, the API retries on a suitable fallback model.
            betas=[_FALLBACK_BETA],
            fallbacks="default",
        )
        self._raise_on_refusal(response)
        # Keep every block, thinking included: later turns must replay them unchanged.
        return {
            "role": "assistant",
            "content": [b.model_dump(mode="json", exclude_none=True) for b in response.content],
        }

    async def diagnose(self, system: str, messages: list[Message]) -> Diagnosis:
        response = await self._client.beta.messages.parse(
            model=self._model,
            max_tokens=16000,
            system=system,
            messages=cast("list[BetaMessageParam]", messages),
            output_format=Diagnosis,
            output_config=self._output_config,
            betas=[_FALLBACK_BETA],
            fallbacks="default",
        )
        self._raise_on_refusal(response)
        if response.parsed_output is None:
            raise RuntimeError(f"No diagnosis returned (stop_reason={response.stop_reason}).")
        return response.parsed_output

    @staticmethod
    def _raise_on_refusal(response: Any) -> None:
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            raise ModelRefusedError(f"The model declined this request (category: {category}).")


PROVIDERS = ("claude", "gemini", "groq")


def triage_model_from_env() -> TriageModel:
    """The model named by TRIAGE_PROVIDER ("claude" by default) and TRIAGE_MODEL."""
    provider = os.environ.get("TRIAGE_PROVIDER", "claude").strip().lower()
    if provider == "claude":
        return ClaudeTriageModel()
    if provider in ("gemini", "groq"):
        from pipeline_triage import langchain_llm

        build = langchain_llm.gemini_model if provider == "gemini" else langchain_llm.groq_model
        return build(os.environ.get("TRIAGE_MODEL"))
    raise ValueError(f"Unknown TRIAGE_PROVIDER {provider!r}. Use one of: {', '.join(PROVIDERS)}.")

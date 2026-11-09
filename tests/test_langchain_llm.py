"""LangChainTriageModel: history translation, thought signatures, structured output, refusals.

A fake LangChain chat model stands in for Gemini and Groq, so these tests need no API
key. The live tests call the real APIs and only run when TRIAGE_TEST_LIVE names the
provider, for example TRIAGE_TEST_LIVE=gemini,groq.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from typing import Any

import groq
import httpx
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import InMemorySaver
from mcp import Client
from pydantic import Field

from pipeline_triage.agent import build_graph, diagnosis_of, initial_state, thread_config
from pipeline_triage.langchain_llm import (
    LangChainTriageModel,
    gemini_model,
    groq_model,
    to_langchain,
)
from pipeline_triage.llm import ClaudeTriageModel, ModelRefusedError, triage_model_from_env
from pipeline_triage.mcp_server import build_server
from pipeline_triage.store import RunStore
from pipeline_triage.toolbox import McpToolbox
from pipeline_triage.triage_cli import _explain

pytestmark = pytest.mark.anyio

SIGNATURES = "__gemini_function_call_thought_signatures__"
TOOL = {
    "name": "get_task_log",
    "description": "",
    "input_schema": {"type": "object", "properties": {"run_id": {"type": "string"}}},
}
DIAGNOSIS = {
    "run_id": "r1",
    "root_cause": "out_of_memory",
    "summary": "OOMKilled.",
    "evidence": ["pod OOMKilled"],
    "runbook_id": "worker-oom",
    "fix_steps": ["Raise the memory limit"],
    "related_run_ids": [],
    "confidence": "high",
}


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeChat(BaseChatModel):
    """Returns scripted replies and records what it was sent."""

    replies: list[Any]
    """AIMessages to return, or exceptions to raise. Any, so pydantic leaves them as-is."""
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: str | None = None, **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.bound_tools.extend(tools)
        return self


def _model(*replies: AIMessage | Exception) -> tuple[LangChainTriageModel, FakeChat]:
    fake = FakeChat(replies=list(replies))
    return LangChainTriageModel(fake, structured_method="function_calling"), fake


def _tool_call_reply() -> AIMessage:
    return AIMessage(
        content=[],
        tool_calls=[{"name": "get_task_log", "args": {"run_id": "r1"}, "id": "c1"}],
        additional_kwargs={SIGNATURES: {"c1": "c2lnbmF0dXJl"}},
    )


async def test_next_turn_returns_claude_format_and_keeps_native_parts() -> None:
    model, fake = _model(_tool_call_reply())

    reply = await model.next_turn("system", [{"role": "user", "content": "hi"}], [TOOL])

    assert reply["content"] == [
        {"type": "tool_use", "id": "c1", "name": "get_task_log", "input": {"run_id": "r1"}}
    ]
    assert reply["native"]["additional_kwargs"][SIGNATURES] == {"c1": "c2lnbmF0dXJl"}
    assert fake.bound_tools == [TOOL]
    assert isinstance(fake.seen[0][0], SystemMessage) and fake.seen[0][0].content == "system"


async def test_history_replays_signatures_and_tool_errors() -> None:
    model, _ = _model(_tool_call_reply())
    assistant = await model.next_turn("system", [{"role": "user", "content": "hi"}], [TOOL])
    result = {"type": "tool_result", "tool_use_id": "c1", "content": "boom", "is_error": True}

    converted = to_langchain(
        "system",
        [{"role": "user", "content": "hi"}, assistant, {"role": "user", "content": [result]}],
    )

    system, human, ai, tool = converted
    assert isinstance(system, SystemMessage) and isinstance(human, HumanMessage)
    assert isinstance(ai, AIMessage)
    assert ai.additional_kwargs[SIGNATURES] == {"c1": "c2lnbmF0dXJl"}
    assert [c["id"] for c in ai.tool_calls] == ["c1"]
    assert isinstance(tool, ToolMessage)
    assert (tool.tool_call_id, tool.name, tool.status, tool.content) == (
        "c1",
        "get_task_log",
        "error",
        "boom",
    )


async def test_diagnose_returns_structured_output() -> None:
    # With function calling, the schema is bound as a tool and the call is parsed.
    model, _ = _model(
        AIMessage(content="", tool_calls=[{"name": "Diagnosis", "args": DIAGNOSIS, "id": "d"}])
    )

    diagnosis = await model.diagnose("system", [{"role": "user", "content": "diagnose"}])

    assert diagnosis.root_cause == "out_of_memory"
    assert diagnosis.runbook_id == "worker-oom"


def _rejected_call() -> groq.BadRequestError:
    body = {"error": {"message": "`/since`: '2026-10-28' is not valid date-time"}}
    body["error"]["code"] = "tool_use_failed"
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com"))
    return groq.BadRequestError("Error code: 400", response=response, body=body)


async def test_rejected_tool_call_is_retried_with_the_reason() -> None:
    model, fake = _model(_rejected_call(), _tool_call_reply())

    reply = await model.next_turn("system", [{"role": "user", "content": "hi"}], [TOOL])

    assert reply["content"][0]["name"] == "get_task_log"
    retry_prompt = fake.seen[1]
    assert isinstance(retry_prompt[-1], HumanMessage)
    assert "not valid date-time" in str(retry_prompt[-1].content)
    # The note only goes with the retry; the first request had no note.
    assert len(fake.seen[0]) == len(retry_prompt) - 1


async def test_rejected_tool_calls_give_up_after_the_repairs() -> None:
    model, fake = _model(_rejected_call(), _rejected_call(), _rejected_call())

    with pytest.raises(groq.BadRequestError):
        await model.next_turn("system", [{"role": "user", "content": "hi"}], [TOOL])
    assert len(fake.seen) == 3


@pytest.mark.parametrize("reason", ["SAFETY", "content_filter"])
async def test_safety_stop_raises(reason: str) -> None:
    model, _ = _model(AIMessage(content="", response_metadata={"finish_reason": reason}))

    with pytest.raises(ModelRefusedError, match=reason):
        await model.next_turn("system", [{"role": "user", "content": "hi"}], [])


def test_provider_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.delenv("TRIAGE_MODEL", raising=False)

    monkeypatch.setenv("TRIAGE_PROVIDER", "Gemini")
    assert isinstance(triage_model_from_env(), LangChainTriageModel)
    monkeypatch.setenv("TRIAGE_PROVIDER", "groq")
    assert isinstance(triage_model_from_env(), LangChainTriageModel)
    monkeypatch.delenv("TRIAGE_PROVIDER")
    assert isinstance(triage_model_from_env(), ClaudeTriageModel)

    monkeypatch.setenv("TRIAGE_PROVIDER", "llama")
    with pytest.raises(ValueError, match="Unknown TRIAGE_PROVIDER"):
        triage_model_from_env()


def _groq_error(status: int) -> groq.APIStatusError:
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.groq.com"))
    error_class = {401: groq.AuthenticationError, 429: groq.RateLimitError}[status]
    return error_class(f"Error code: {status}", response=response, body=None)


def test_cli_explains_provider_errors() -> None:
    assert "GROQ_API_KEY" in (_explain(_groq_error(401)) or "")
    assert "tokens per minute" in (_explain(_groq_error(429)) or "")
    assert "GROQ_API_KEY" in (_explain(groq.GroqError("The api_key client option ...")) or "")
    assert "TRIAGE_PROVIDER" in (_explain(ValueError("Unknown TRIAGE_PROVIDER 'x'.")) or "")
    assert _explain(RuntimeError("something else")) is None


@pytest.fixture
async def toolbox(store: RunStore) -> AsyncIterator[McpToolbox]:
    async with Client(build_server(store)) as client:
        yield await McpToolbox(client).load()


@pytest.mark.live
@pytest.mark.parametrize("provider", ["gemini", "groq"])
async def test_live_model_triages_an_incident(toolbox: McpToolbox, provider: str) -> None:
    if provider not in os.environ.get("TRIAGE_TEST_LIVE", "").split(","):
        pytest.skip(f"set TRIAGE_TEST_LIVE={provider} and its API key to call the real API")
    model = gemini_model() if provider == "gemini" else groq_model()
    run_id = "vendor_payments_sync__2026-10-29"
    graph = build_graph(model, toolbox, InMemorySaver())
    config = thread_config(run_id)

    await graph.ainvoke(initial_state(run_id), config)

    snapshot = graph.get_state(config)
    assert snapshot.next == ("approve",)
    assert snapshot.values["tool_rounds"] >= 1  # it investigated before answering
    assert diagnosis_of(snapshot.values).root_cause == "credential_expiry"

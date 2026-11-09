"""GeminiTriageModel: history translation, thought signatures, structured output, refusals.

A fake LangChain chat model stands in for Gemini, so these tests need no API key.
The last test calls the real API and only runs with TRIAGE_TEST_GEMINI=1.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence
from typing import Any

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
from pipeline_triage.gemini import GeminiTriageModel, to_langchain
from pipeline_triage.llm import ClaudeTriageModel, ModelRefusedError, triage_model_from_env
from pipeline_triage.mcp_server import build_server
from pipeline_triage.store import RunStore
from pipeline_triage.toolbox import McpToolbox

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


class FakeGemini(BaseChatModel):
    """Returns scripted replies and records what it was sent."""

    replies: list[AIMessage]
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-gemini"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: str | None = None, **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.bound_tools.extend(tools)
        return self


def _tool_call_reply() -> AIMessage:
    return AIMessage(
        content=[],
        tool_calls=[{"name": "get_task_log", "args": {"run_id": "r1"}, "id": "c1"}],
        additional_kwargs={SIGNATURES: {"c1": "c2lnbmF0dXJl"}},
    )


async def test_next_turn_returns_claude_format_and_keeps_signatures() -> None:
    fake = FakeGemini(replies=[_tool_call_reply()])

    reply = await GeminiTriageModel(chat=fake).next_turn(
        "system", [{"role": "user", "content": "hi"}], [TOOL]
    )

    assert reply["content"] == [
        {"type": "tool_use", "id": "c1", "name": "get_task_log", "input": {"run_id": "r1"}}
    ]
    assert reply["gemini"]["additional_kwargs"][SIGNATURES] == {"c1": "c2lnbmF0dXJl"}
    assert fake.bound_tools == [TOOL]
    assert isinstance(fake.seen[0][0], SystemMessage) and fake.seen[0][0].content == "system"


async def test_history_replays_signatures_and_tool_errors() -> None:
    fake = FakeGemini(replies=[_tool_call_reply()])
    assistant = await GeminiTriageModel(chat=fake).next_turn(
        "system", [{"role": "user", "content": "hi"}], [TOOL]
    )
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
    # BaseChatModel's structured output binds the schema as a tool and parses the call.
    reply = AIMessage(content="", tool_calls=[{"name": "Diagnosis", "args": DIAGNOSIS, "id": "d"}])
    model = GeminiTriageModel(chat=FakeGemini(replies=[reply]))

    diagnosis = await model.diagnose("system", [{"role": "user", "content": "diagnose"}])

    assert diagnosis.root_cause == "out_of_memory"
    assert diagnosis.runbook_id == "worker-oom"


async def test_safety_stop_raises() -> None:
    reply = AIMessage(content="", response_metadata={"finish_reason": "SAFETY"})
    model = GeminiTriageModel(chat=FakeGemini(replies=[reply]))

    with pytest.raises(ModelRefusedError, match="SAFETY"):
        await model.next_turn("system", [{"role": "user", "content": "hi"}], [])


def test_provider_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("TRIAGE_PROVIDER", "Gemini")
    assert isinstance(triage_model_from_env(), GeminiTriageModel)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.delenv("TRIAGE_PROVIDER")
    assert isinstance(triage_model_from_env(), ClaudeTriageModel)

    monkeypatch.setenv("TRIAGE_PROVIDER", "llama")
    with pytest.raises(ValueError, match="Unknown TRIAGE_PROVIDER"):
        triage_model_from_env()


@pytest.fixture
async def toolbox(store: RunStore) -> AsyncIterator[McpToolbox]:
    async with Client(build_server(store)) as client:
        yield await McpToolbox(client).load()


@pytest.mark.gemini
async def test_real_gemini_triages_an_incident(toolbox: McpToolbox) -> None:
    if os.environ.get("TRIAGE_TEST_GEMINI") != "1":
        pytest.skip("set TRIAGE_TEST_GEMINI=1 and GEMINI_API_KEY to call the real Gemini API")
    run_id = "vendor_payments_sync__2026-10-29"
    graph = build_graph(GeminiTriageModel(), toolbox, InMemorySaver())
    config = thread_config(run_id)

    await graph.ainvoke(initial_state(run_id), config)

    snapshot = graph.get_state(config)
    assert snapshot.next == ("approve",)
    assert snapshot.values["tool_rounds"] >= 1  # it investigated before answering
    assert diagnosis_of(snapshot.values).root_cause == "credential_expiry"

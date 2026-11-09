"""The triage graph end to end, with a scripted model in place of Claude.

The scripted model makes fixed tool calls, so these tests cover the LangGraph
wiring, the real MCP server and the approval pause without an API key.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from mcp import Client

from pipeline_triage.agent import (
    build_graph,
    diagnosis_of,
    initial_state,
    system_prompt,
    thread_config,
)
from pipeline_triage.diagnosis import Diagnosis
from pipeline_triage.llm import Message
from pipeline_triage.mcp_server import build_server
from pipeline_triage.store import RunStore
from pipeline_triage.toolbox import McpToolbox

pytestmark = pytest.mark.anyio

RUN_ID = "vendor_payments_sync__2026-10-29"

DIAGNOSIS = Diagnosis(
    run_id=RUN_ID,
    root_cause="credential_expiry",
    summary="The vendor OAuth client secret expired, so the token refresh returned 401.",
    evidence=["invalid_client: Client secret has expired"],
    runbook_id="expired-credentials",
    fix_steps=["Rotate the client secret", "Update the Airflow connection", "Rerun the run"],
    related_run_ids=[],
    confidence="high",
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class ScriptedModel:
    """Plays back tool calls, one turn each, then a final text answer."""

    def __init__(self, calls: list[tuple[str, dict[str, Any]]], repeat: bool = False) -> None:
        self._calls = calls
        self._repeat = repeat
        self.turns = 0
        self.diagnose_messages: list[Message] = []

    async def next_turn(
        self, system: str, messages: list[Message], tools: list[dict[str, Any]]
    ) -> Message:
        assert {t["name"] for t in tools} >= {"get_run_status", "get_task_log"}
        index = self.turns % len(self._calls) if self._repeat else self.turns
        self.turns += 1
        if index >= len(self._calls):
            return {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}
        name, args = self._calls[index]
        block = {"type": "tool_use", "id": f"call_{self.turns}", "name": name, "input": args}
        return {"role": "assistant", "content": [block]}

    async def diagnose(self, system: str, messages: list[Message]) -> Diagnosis:
        self.diagnose_messages = messages
        return DIAGNOSIS


@pytest.fixture
async def toolbox(store: RunStore) -> AsyncIterator[McpToolbox]:
    async with Client(build_server(store)) as client:
        yield await McpToolbox(client).load()


def _tool_results(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        block
        for m in state["messages"]
        if m["role"] == "user" and isinstance(m["content"], list)
        for block in m["content"]
    ]


async def test_investigates_through_mcp_then_waits_for_approval(toolbox: McpToolbox) -> None:
    model = ScriptedModel(
        [
            ("get_run_status", {"run_id": RUN_ID}),
            ("get_task_log", {"run_id": RUN_ID, "task_id": "extract_vendor_api"}),
        ]
    )
    graph = build_graph(model, toolbox, InMemorySaver())
    config = thread_config(RUN_ID)

    await graph.ainvoke(initial_state(RUN_ID), config)

    snapshot = graph.get_state(config)
    assert snapshot.next == ("approve",)  # paused for a human
    assert snapshot.interrupts[0].value["diagnosis"]["root_cause"] == "credential_expiry"
    results = _tool_results(snapshot.values)
    assert len(results) == 2 and not any(r["is_error"] for r in results)
    assert "Client secret has expired" in results[1]["content"]
    assert diagnosis_of(snapshot.values) == DIAGNOSIS
    # The diagnose call sees the whole investigation plus the closing request.
    assert "structured output" in model.diagnose_messages[-1]["content"]


@pytest.mark.parametrize("approved", [True, False])
async def test_resume_records_the_human_decision(toolbox: McpToolbox, approved: bool) -> None:
    graph = build_graph(ScriptedModel([]), toolbox, InMemorySaver())
    config = thread_config("t")
    await graph.ainvoke(initial_state(RUN_ID), config)

    final = await graph.ainvoke(Command(resume={"approved": approved, "note": "ok"}), config)

    assert final["decision"] == {"approved": approved, "note": "ok"}
    assert graph.get_state(config).next == ()


async def test_tool_errors_are_returned_to_the_model(toolbox: McpToolbox) -> None:
    model = ScriptedModel([("get_run_status", {"run_id": "no_such_run"})])
    graph = build_graph(model, toolbox, InMemorySaver())
    config = thread_config("t")

    await graph.ainvoke(initial_state("no_such_run"), config)

    [result] = _tool_results(graph.get_state(config).values)
    assert result["is_error"]
    assert "Unknown run_id" in result["content"]


async def test_step_limit_ends_the_investigation_cleanly(toolbox: McpToolbox) -> None:
    model = ScriptedModel([("get_failed_runs", {})], repeat=True)
    graph = build_graph(model, toolbox, InMemorySaver(), max_tool_rounds=3)
    config = thread_config("t")

    await graph.ainvoke(initial_state(RUN_ID), config)

    state = graph.get_state(config).values
    assert state["tool_rounds"] == 3
    assert model.turns == 4
    # The fourth, unanswered tool call gets an error result, so the history stays valid.
    last_results = model.diagnose_messages[-2]["content"]
    assert last_results[0]["is_error"]
    assert "step limit" in last_results[0]["content"]


def test_diagnosis_markdown() -> None:
    text = DIAGNOSIS.to_markdown()
    assert text.startswith(f"## Triage: {RUN_ID}")
    assert "1. Rotate the client secret" in text
    assert "**Runbook:** expired-credentials" in text


def test_system_prompt_names_runbook_search_only_when_offered() -> None:
    assert "search_runbooks" not in system_prompt({"get_run_status", "get_task_log"})
    assert "search_runbooks" in system_prompt({"get_run_status", "search_runbooks"})

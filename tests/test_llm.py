"""ClaudeTriageModel against a fake HTTP transport: request shape and response handling."""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient

from pipeline_triage.llm import ClaudeTriageModel, ModelRefusedError

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _message(content: list[dict[str, Any]], stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _model(reply: dict[str, Any], seen: list[httpx2.Request]) -> ClaudeTriageModel:
    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=reply)

    http = DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))
    client = AsyncAnthropic(api_key="test", http_client=http, max_retries=0)
    return ClaudeTriageModel(model="claude-opus-5-5", effort="high", client=client)


async def test_next_turn_sends_tools_effort_and_fallbacks() -> None:
    seen: list[httpx2.Request] = []
    tool_use = {"type": "tool_use", "id": "t1", "name": "get_failed_runs", "input": {}}
    model = _model(_message([tool_use], "tool_use"), seen)

    reply = await model.next_turn(
        "system",
        [{"role": "user", "content": "hi"}],
        [
            {
                "name": "get_failed_runs",
                "description": "",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
    )

    body = json.loads(seen[0].content)
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"] == {"effort": "high"}
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in seen[0].headers["anthropic-beta"]
    assert body["tools"][0]["name"] == "get_failed_runs"
    assert reply == {"role": "assistant", "content": [tool_use]}


async def test_refusal_raises() -> None:
    model = _model(_message([], "refusal"), [])
    with pytest.raises(ModelRefusedError):
        await model.next_turn("system", [{"role": "user", "content": "hi"}], [])


async def test_diagnose_parses_structured_output() -> None:
    diagnosis = {
        "run_id": "r1",
        "root_cause": "out_of_memory",
        "summary": "OOMKilled.",
        "evidence": ["pod OOMKilled"],
        "runbook_id": "worker-oom",
        "fix_steps": ["Raise the memory limit"],
        "related_run_ids": [],
        "confidence": "high",
    }
    seen: list[httpx2.Request] = []
    model = _model(_message([{"type": "text", "text": json.dumps(diagnosis)}]), seen)

    result = await model.diagnose("system", [{"role": "user", "content": "diagnose"}])

    assert result.root_cause == "out_of_memory"
    assert json.loads(seen[0].content)["output_config"]["format"]["type"] == "json_schema"

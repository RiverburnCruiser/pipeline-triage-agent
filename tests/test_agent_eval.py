"""The agent eval harness, with scripted models that are always right or always wrong."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any

import pytest
from mcp import Client

from pipeline_triage.agent_eval import evaluate, format_report
from pipeline_triage.diagnosis import Diagnosis
from pipeline_triage.llm import Message
from pipeline_triage.mcp_server import build_server
from pipeline_triage.retrieval_eval import IncidentLabel, load_labels
from pipeline_triage.store import RunStore
from pipeline_triage.toolbox import McpToolbox

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def toolbox(store: RunStore) -> AsyncIterator[McpToolbox]:
    async with Client(build_server(store)) as client:
        yield await McpToolbox(client).load()


def _run_id(messages: list[Message]) -> str:
    match = re.search(r"run (\S+)\.", messages[0]["content"])
    assert match
    return match.group(1)


class LabelModel:
    """Checks the run once, then answers with the label (or a wrong answer)."""

    def __init__(self, labels: dict[str, IncidentLabel], right: bool = True) -> None:
        self._labels = labels
        self._right = right

    async def next_turn(
        self, system: str, messages: list[Message], tools: list[dict[str, Any]]
    ) -> Message:
        if len(messages) == 1:
            call = {"run_id": _run_id(messages)}
            block = {"type": "tool_use", "id": "c1", "name": "get_run_status", "input": call}
            return {"role": "assistant", "content": [block]}
        return {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}

    async def diagnose(self, system: str, messages: list[Message]) -> Diagnosis:
        run_id = _run_id(messages)
        label = self._labels[run_id]
        if run_id.startswith("vendor"):
            raise RuntimeError("quota used up")
        return Diagnosis(
            run_id=run_id,
            root_cause=label.root_cause if self._right else "unknown",  # type: ignore[arg-type]
            summary="scripted",
            evidence=[],
            runbook_id=label.runbook_id if self._right else None,
            fix_steps=[],
            related_run_ids=label.related_run_ids if self._right else [],
            confidence="high",
        )


async def test_scores_each_incident_against_its_label(toolbox: McpToolbox) -> None:
    labels = load_labels()
    report = await evaluate(LabelModel(labels), toolbox, labels, "scripted/right")

    assert len(report.incidents) == 5
    by_run = {s.run_id: s for s in report.incidents}
    errored = by_run.pop("vendor_payments_sync__2026-10-29")
    assert errored.error == "RuntimeError: quota used up"
    assert not errored.root_cause_ok
    assert all(
        s.root_cause_ok and s.related_ok and s.tools_called == ["get_run_status"]
        for s in by_run.values()
    )
    # No DATABASE_URL in the test server, so the runbook is not scored.
    assert not report.runbook_search
    assert report.correct_rate == pytest.approx(0.8)
    assert report.related_accuracy == 1.0

    text = format_report(report)
    assert "**Correct: 80%** of 5 incidents" in text
    assert "error: RuntimeError: quota used up" in text


async def test_wrong_answers_score_zero(toolbox: McpToolbox) -> None:
    labels = load_labels()
    report = await evaluate(LabelModel(labels, right=False), toolbox, labels, "scripted/wrong")
    assert report.correct_rate == 0.0
    assert report.related_accuracy == 0.0
    assert "unknown (**no**)" in format_report(report)

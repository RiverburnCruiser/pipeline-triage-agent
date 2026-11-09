"""Score the whole triage agent on the labelled sample incidents.

Each incident runs the full graph (the configured model, the real MCP server, runbook
search when DATABASE_URL is set) up to the approval pause. The diagnosis is then
checked against labels.json:

    root cause     matches the root_cause label
    runbook        matches the runbook_id label (only scored with runbook search)
    related runs   names the expected upstream runs (only for knock-on incidents)

An incident is correct when its root cause and runbook are both right. Errors, such as
a used-up free-tier quota, count as wrong and are reported.

    pipeline-triage-eval-agent [--min-correct 0.8] [--json report.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from mcp import Client
from pydantic import BaseModel

from pipeline_triage.agent import build_graph, diagnosis_of, initial_state, thread_config
from pipeline_triage.errors import explain_error
from pipeline_triage.llm import DEFAULT_MODEL, TriageModel, triage_model_from_env
from pipeline_triage.retrieval_eval import IncidentLabel, load_labels
from pipeline_triage.toolbox import McpToolbox, stdio_server


class IncidentScore(BaseModel):
    run_id: str
    expected_root_cause: str
    root_cause: str | None = None
    expected_runbook: str
    runbook_id: str | None = None
    expected_related: list[str] = []
    related_run_ids: list[str] = []
    tool_calls: int = 0
    seconds: float = 0.0
    error: str | None = None

    @property
    def root_cause_ok(self) -> bool:
        return self.root_cause == self.expected_root_cause

    @property
    def runbook_ok(self) -> bool:
        return self.runbook_id == self.expected_runbook

    @property
    def related_ok(self) -> bool:
        return set(self.expected_related) <= set(self.related_run_ids)


class AgentEvalReport(BaseModel):
    model: str
    runbook_search: bool
    incidents: list[IncidentScore]

    def correct(self, score: IncidentScore) -> bool:
        return score.root_cause_ok and (score.runbook_ok or not self.runbook_search)

    def _share(self, values: list[bool]) -> float:
        return sum(values) / len(values) if values else 0.0

    @property
    def correct_rate(self) -> float:
        return self._share([self.correct(s) for s in self.incidents])

    @property
    def root_cause_accuracy(self) -> float:
        return self._share([s.root_cause_ok for s in self.incidents])

    @property
    def runbook_accuracy(self) -> float | None:
        return self._share([s.runbook_ok for s in self.incidents]) if self.runbook_search else None

    @property
    def related_accuracy(self) -> float:
        return self._share([s.related_ok for s in self.incidents if s.expected_related])

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "runbook_search": self.runbook_search,
            "correct": self.correct_rate,
            "root_cause_accuracy": self.root_cause_accuracy,
            "runbook_accuracy": self.runbook_accuracy,
            "related_accuracy": self.related_accuracy,
            "mean_tool_calls": self._mean([s.tool_calls for s in self.incidents]),
            "mean_seconds": self._mean([s.seconds for s in self.incidents]),
            "errors": sum(1 for s in self.incidents if s.error),
        }

    @staticmethod
    def _mean(values: list[float] | list[int]) -> float:
        return sum(values) / len(values) if values else 0.0


def _count_tool_calls(state: dict[str, Any]) -> int:
    return sum(
        1
        for message in state.get("messages", [])
        if message["role"] == "assistant"
        for block in message["content"]
        if block.get("type") == "tool_use"
    )


async def evaluate(
    model: TriageModel,
    toolbox: McpToolbox,
    labels: dict[str, IncidentLabel],
    model_name: str,
) -> AgentEvalReport:
    """Triage every labelled incident in turn (in turn, to stay inside rate limits)."""
    graph = build_graph(model, toolbox, InMemorySaver())
    scores = []
    for run_id, label in labels.items():
        score = IncidentScore(
            run_id=run_id,
            expected_root_cause=label.root_cause,
            expected_runbook=label.runbook_id,
            expected_related=label.related_run_ids,
        )
        triage_id = f"eval-{uuid.uuid4().hex}"
        started = time.perf_counter()
        try:
            state = await graph.ainvoke(initial_state(run_id), thread_config(triage_id))
            diagnosis = diagnosis_of(state)
            score.root_cause = diagnosis.root_cause
            score.runbook_id = diagnosis.runbook_id
            score.related_run_ids = diagnosis.related_run_ids
            score.tool_calls = _count_tool_calls(state)
        except Exception as error:
            score.error = explain_error(error) or f"{type(error).__name__}: {error}"
        score.seconds = round(time.perf_counter() - started, 1)
        scores.append(score)
    return AgentEvalReport(
        model=model_name, runbook_search="search_runbooks" in toolbox.names, incidents=scores
    )


def _mark(ok: bool) -> str:
    return "yes" if ok else "**no**"


def format_report(report: AgentEvalReport) -> str:
    """A Markdown report, readable in a terminal and in a GitHub job summary."""
    search = "with runbook search" if report.runbook_search else "without runbook search"
    lines = [
        f"### Agent eval: `{report.model}`, {search}",
        "",
        "| Incident | Root cause | Runbook | Related runs | Tool calls | Seconds |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in report.incidents:
        if s.error:
            lines.append(f"| `{s.run_id}` | error: {s.error.splitlines()[0]} | | | | {s.seconds} |")
            continue
        cause = f"{s.root_cause} ({_mark(s.root_cause_ok)})"
        runbook = f"{s.runbook_id} ({_mark(s.runbook_ok)})" if report.runbook_search else "-"
        related = _mark(s.related_ok) if s.expected_related else "-"
        lines.append(
            f"| `{s.run_id}` | {cause} | {runbook} | {related} | {s.tool_calls} | {s.seconds} |"
        )
    summary = report.summary()
    runbook = summary["runbook_accuracy"]
    lines += [
        "",
        f"**Correct: {summary['correct']:.0%}** of {len(report.incidents)} incidents"
        f" (root cause {summary['root_cause_accuracy']:.0%}"
        + (f", runbook {runbook:.0%}" if runbook is not None else "")
        + f", related runs {summary['related_accuracy']:.0%})."
        f" Mean {summary['mean_tool_calls']:.1f} tool calls and"
        f" {summary['mean_seconds']:.0f} s per triage.",
    ]
    if summary["errors"]:
        lines.append(f"{summary['errors']} incident(s) failed with an error.")
    return "\n".join(lines)


def configured_model() -> str:
    """The provider and model the environment selects, for the report."""
    from pipeline_triage.langchain_llm import DEFAULT_MODELS

    provider = os.environ.get("TRIAGE_PROVIDER", "claude").strip().lower()
    default = DEFAULT_MODEL if provider == "claude" else DEFAULT_MODELS.get(provider, "")
    return f"{provider}/{os.environ.get('TRIAGE_MODEL') or default}"


async def _run() -> AgentEvalReport:
    model = triage_model_from_env()
    async with Client(stdio_server()) as client:
        toolbox = await McpToolbox(client).load()
        return await evaluate(model, toolbox, load_labels(), configured_model())


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the triage agent on the sample incidents.")
    parser.add_argument(
        "--min-correct",
        type=float,
        default=0.0,
        help="Exit with status 1 when fewer incidents than this share are correct (0 to 1).",
    )
    parser.add_argument("--json", type=Path, help="Also write the report as JSON to this file.")
    args = parser.parse_args()

    report = asyncio.run(_run())
    text = format_report(report)
    print(text)
    if summary_file := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    if args.json:
        payload = {"summary": report.summary(), **report.model_dump()}
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    if report.correct_rate < args.min_correct:
        sys.exit(f"Correct {report.correct_rate:.0%} is below the minimum {args.min_correct:.0%}.")

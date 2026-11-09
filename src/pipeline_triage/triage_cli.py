"""Run the triage agent from a terminal.

    pipeline-triage-agent                 triage the most recent failed run
    pipeline-triage-agent RUN_ID          triage a specific run

The agent starts the MCP server as a subprocess and talks to it over stdio, the same
way Claude Code does. Needs ANTHROPIC_API_KEY, or for a free tier TRIAGE_PROVIDER=gemini
with GEMINI_API_KEY, or TRIAGE_PROVIDER=groq with GROQ_API_KEY. Set DATABASE_URL to give
the agent runbook search.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from mcp import Client

from pipeline_triage.agent import (
    Decision,
    build_graph,
    diagnosis_of,
    initial_state,
    thread_config,
)
from pipeline_triage.errors import explain_error
from pipeline_triage.llm import triage_model_from_env
from pipeline_triage.toolbox import McpToolbox, stdio_server


def _print_progress(update: dict[str, Any]) -> None:
    for node, change in update.items():
        if node == "investigate" and change:
            for block in change["messages"][-1]["content"]:
                if block.get("type") == "tool_use":
                    args = ", ".join(f"{k}={v!r}" for k, v in block["input"].items())
                    print(f"  -> {block['name']}({args})")
        elif node == "diagnose":
            print("  -> writing diagnosis")


def _ask_for_decision() -> Decision:
    answer = input("\nApprove this fix? [y/N] ").strip().lower()
    note = input("Note for the record (optional): ").strip()
    return {"approved": answer in {"y", "yes"}, "note": note}


async def _latest_failed_run(toolbox: McpToolbox) -> str:
    runs = await toolbox.failed_runs(limit=1)
    if not runs:
        sys.exit("No failed runs found.")
    return str(runs[0]["run_id"])


async def run(run_id: str | None) -> None:
    model = triage_model_from_env()
    async with Client(stdio_server()) as client:
        toolbox = await McpToolbox(client).load()
        run_id = run_id or await _latest_failed_run(toolbox)
        if "search_runbooks" not in toolbox.names:
            print("Note: DATABASE_URL is not set, so the agent has no runbook search.")
        print(f"Triaging {run_id}")

        graph = build_graph(model, toolbox, InMemorySaver())
        config = thread_config(run_id)
        async for update in graph.astream(initial_state(run_id), config, stream_mode="updates"):
            _print_progress(update)

        state = graph.get_state(config)
        print("\n" + diagnosis_of(state.values).to_markdown())

        decision = _ask_for_decision()
        await graph.ainvoke(Command(resume=decision), config)
        verdict = "Approved" if decision["approved"] else "Rejected"
        print(f"\n{verdict}." + (f" Note: {decision['note']}" if decision["note"] else ""))
        print("Nothing was executed: approved fixes are recorded, not run.")


def _first_error(group: BaseExceptionGroup[BaseException]) -> BaseException:
    error: BaseException = group
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    return error


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage a failed pipeline run with an AI agent.")
    parser.add_argument("run_id", nargs="?", help="Run to triage (default: most recent failure)")
    args = parser.parse_args()
    # The MCP client runs in a task group, so errors arrive wrapped in exception groups.
    try:
        asyncio.run(run(args.run_id))
    except* Exception as group:
        message = explain_error(_first_error(group))
        if message is None:
            raise
        sys.exit(message)

"""Run the triage agent from a terminal.

    pipeline-triage-agent                 triage the most recent failed run
    pipeline-triage-agent RUN_ID          triage a specific run

The agent starts the MCP server as a subprocess and talks to it over stdio, the same
way Claude Code does. Needs ANTHROPIC_API_KEY, or TRIAGE_PROVIDER=gemini and
GEMINI_API_KEY for Gemini's free tier. Set DATABASE_URL to give the agent runbook search.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any

from anthropic import AuthenticationError
from langchain_core.exceptions import ModelError
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from mcp import Client, StdioServerParameters

from pipeline_triage.agent import (
    Decision,
    build_graph,
    diagnosis_of,
    initial_state,
    thread_config,
)
from pipeline_triage.llm import ModelRefusedError, triage_model_from_env
from pipeline_triage.toolbox import McpToolbox


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
    result = await toolbox.call("get_failed_runs", {"limit": 1})
    runs = (result.data or {}).get("result", [])
    if not runs:
        sys.exit("No failed runs found.")
    return str(runs[0]["run_id"])


async def run(run_id: str | None) -> None:
    model = triage_model_from_env()
    server = StdioServerParameters(
        command=sys.executable, args=["-m", "pipeline_triage.mcp_server"], env=dict(os.environ)
    )
    async with Client(server) as client:
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
    except* AuthenticationError:
        sys.exit("Claude rejected the credentials. Check ANTHROPIC_API_KEY (see README).")
    except* TypeError as group:
        # Raised by the SDK when it finds no credentials at all.
        if "authentication method" in str(_first_error(group)):
            sys.exit("No Claude credentials found. Set ANTHROPIC_API_KEY (see README).")
        raise
    except* ModelRefusedError as group:
        sys.exit(str(_first_error(group)))
    except* ModelError as group:
        # LangChain's base class for provider errors: here, Gemini's.
        error = _first_error(group)
        message = str(error)
        if "API key not valid" in message:
            sys.exit("Gemini rejected the API key. Check GEMINI_API_KEY (see README).")
        hints = {
            429: "This model's quota is used up. The free tier limits requests per minute and"
            " per day for each model, so wait, or switch with TRIAGE_MODEL (see README).",
            503: "The model is overloaded. Try again, or use another one, for example"
            " TRIAGE_MODEL=gemini-3.5-flash-lite.",
        }
        # The HTTP status is on the underlying google-genai error.
        code = getattr(error, "code", None) or getattr(error.__cause__, "code", None)
        hint = hints.get(code, "") if isinstance(code, int) else ""
        sys.exit(f"Gemini request failed: {message.splitlines()[0][:300]}\n{hint}".strip())
    except* ValueError as group:
        # Raised when TRIAGE_PROVIDER is unknown, or by the Gemini client without a key.
        message = str(_first_error(group))
        if "API key required" in message:
            sys.exit("No Gemini API key found. Set GEMINI_API_KEY (see README).")
        if "TRIAGE_PROVIDER" in message:
            sys.exit(message)
        raise

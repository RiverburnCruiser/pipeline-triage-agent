"""The triage agent as a LangGraph state machine.

    START -> investigate <-> run_tools      the model calls MCP tools to gather evidence
                  |
                  v
               diagnose                     one structured call: root cause, runbook, fix
                  |
                  v
               approve                      pauses (interrupt) until a human decides
                  |
                  v
                 END

The graph only reads: every MCP tool is read-only, and the approval step records a
decision without acting on it. Executing approved fixes is out of scope for now.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from pipeline_triage.diagnosis import Diagnosis
from pipeline_triage.llm import Message, TriageModel
from pipeline_triage.toolbox import McpToolbox

MAX_TOOL_ROUNDS = 12

SYSTEM_PROMPT = """\
You are an on-call data engineer triaging a failed data pipeline run (Airflow-style DAGs).

Investigate before concluding:
1. Check the run's status to find the failed task and any upstream_failed tasks.
2. Read the failed task's log. Logs of tasks that succeeded in the same run can hold
   evidence too, such as warnings about missing input.
3. When the failure could be about data volume, check the row counts of the affected table.
4. When the failure could be caused by another pipeline, check whether related DAGs
   failed around the same time, and say which run is the root and which is a knock-on.
5. If search_runbooks is available, search it with the key error lines and follow the
   matching runbook's diagnosis steps.

Base every claim on what the tools returned. When you have enough evidence, stop
calling tools and write a short summary of what you found."""

DIAGNOSE_PROMPT = """\
Now give your diagnosis of run {run_id} as structured output. Quote evidence from the
tool results. Use the runbook id exactly as search_runbooks returned it, or null if you
did not find a matching runbook. Use low confidence if the evidence is incomplete."""


class Decision(TypedDict):
    approved: bool
    note: str


class TriageState(TypedDict, total=False):
    run_id: str
    messages: list[Message]
    tool_rounds: int
    diagnosis: dict[str, Any]
    decision: Decision


def _tool_calls(message: Message) -> list[dict[str, Any]]:
    return [b for b in message["content"] if b.get("type") == "tool_use"]


def build_graph(
    model: TriageModel,
    toolbox: McpToolbox,
    checkpointer: BaseCheckpointSaver[Any],
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
) -> CompiledStateGraph[TriageState, None, TriageState, TriageState]:
    async def investigate(state: TriageState) -> TriageState:
        reply = await model.next_turn(SYSTEM_PROMPT, state["messages"], toolbox.definitions)
        return {"messages": [*state["messages"], reply]}

    async def run_tools(state: TriageState) -> TriageState:
        results = []
        for call in _tool_calls(state["messages"][-1]):
            result = await toolbox.call(call["name"], call.get("input", {}))
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": result.text,
                    "is_error": result.is_error,
                }
            )
        # All results for one turn go back in a single user message.
        return {
            "messages": [*state["messages"], {"role": "user", "content": results}],
            "tool_rounds": state.get("tool_rounds", 0) + 1,
        }

    def after_investigate(state: TriageState) -> Literal["run_tools", "diagnose"]:
        wants_tools = bool(_tool_calls(state["messages"][-1]))
        under_limit = state.get("tool_rounds", 0) < max_tool_rounds
        return "run_tools" if wants_tools and under_limit else "diagnose"

    async def diagnose(state: TriageState) -> TriageState:
        messages = list(state["messages"])
        if _tool_calls(messages[-1]):
            # Out of tool rounds: answer the pending calls so the history stays valid.
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": c["id"],
                            "content": "Not run: the investigation step limit was reached.",
                            "is_error": True,
                        }
                        for c in _tool_calls(messages[-1])
                    ],
                }
            )
        messages.append({"role": "user", "content": DIAGNOSE_PROMPT.format(run_id=state["run_id"])})
        diagnosis = await model.diagnose(SYSTEM_PROMPT, messages)
        return {"messages": messages, "diagnosis": diagnosis.model_dump()}

    def approve(state: TriageState) -> TriageState:
        # Pauses the graph. The caller resumes it with Command(resume=Decision(...)).
        decision: Decision = interrupt({"diagnosis": state["diagnosis"]})
        return {"decision": decision}

    graph = StateGraph(TriageState)
    graph.add_node("investigate", investigate)
    graph.add_node("run_tools", run_tools)
    graph.add_node("diagnose", diagnose)
    graph.add_node("approve", approve)
    graph.add_edge(START, "investigate")
    graph.add_conditional_edges("investigate", after_investigate)
    graph.add_edge("run_tools", "investigate")
    graph.add_edge("diagnose", "approve")
    graph.add_edge("approve", END)
    return graph.compile(checkpointer=checkpointer)


def initial_state(run_id: str) -> TriageState:
    return {
        "run_id": run_id,
        "messages": [{"role": "user", "content": f"Triage the failed pipeline run {run_id}."}],
        "tool_rounds": 0,
    }


def thread_config(thread_id: str) -> RunnableConfig:
    """Checkpointer key: each triage is its own thread, so it can pause and resume."""
    return {"configurable": {"thread_id": thread_id}}


def diagnosis_of(state: Mapping[str, Any]) -> Diagnosis:
    return Diagnosis.model_validate(state["diagnosis"])

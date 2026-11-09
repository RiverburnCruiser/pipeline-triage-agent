"""Triages as background jobs: start one, follow it, then approve or reject its fix.

A triage runs the graph up to the approval pause, which can take minutes on a free-tier
model, so it runs as an asyncio task and callers poll its status. The checkpointer holds
each paused triage. With PostgreSQL (see api.py) a triage that is waiting for approval
survives a restart of the service.

    investigating -> awaiting_approval -> approved | rejected
          |
          +-> failed (the model or a tool raised, or the service restarted mid-way)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command
from pydantic import BaseModel

from pipeline_triage.agent import build_graph, initial_state, thread_config
from pipeline_triage.diagnosis import Diagnosis
from pipeline_triage.errors import explain_error
from pipeline_triage.llm import TriageModel
from pipeline_triage.toolbox import McpToolbox

logger = logging.getLogger(__name__)

TriageStatus = Literal["investigating", "awaiting_approval", "approved", "rejected", "failed"]


class DecisionIn(BaseModel):
    approved: bool
    note: str = ""


class TriageView(BaseModel):
    triage_id: str
    run_id: str
    status: TriageStatus
    diagnosis: Diagnosis | None = None
    decision: DecisionIn | None = None
    error: str | None = None


class TriageNotFoundError(LookupError):
    pass


class RunNotFoundError(LookupError):
    pass


class NotAwaitingApprovalError(RuntimeError):
    pass


@dataclass
class _Job:
    run_id: str
    task: asyncio.Task[None]
    error: str | None = None


def _describe(error: Exception) -> str:
    return explain_error(error) or f"{type(error).__name__}: {error}"


class TriageService:
    def __init__(
        self,
        model: TriageModel,
        toolbox: McpToolbox,
        checkpointer: BaseCheckpointSaver[Any],
    ) -> None:
        self._toolbox = toolbox
        self._graph = build_graph(model, toolbox, checkpointer)
        self._jobs: dict[str, _Job] = {}
        self._decide_lock = asyncio.Lock()

    @property
    def toolbox(self) -> McpToolbox:
        return self._toolbox

    async def start(self, run_id: str | None = None) -> TriageView:
        """Start triaging `run_id`, or the most recent failed run."""
        if run_id is None:
            latest = await self._toolbox.failed_runs(limit=1)
            if not latest:
                raise RunNotFoundError("There are no failed runs to triage.")
            run_id = str(latest[0]["run_id"])
        elif (await self._toolbox.call("get_run_status", {"run_id": run_id})).is_error:
            raise RunNotFoundError(f"Unknown run {run_id!r}.")

        triage_id = uuid.uuid4().hex
        task = asyncio.create_task(self._investigate(triage_id, run_id))
        self._jobs[triage_id] = _Job(run_id, task)
        return TriageView(triage_id=triage_id, run_id=run_id, status="investigating")

    async def _investigate(self, triage_id: str, run_id: str) -> None:
        try:
            await self._graph.ainvoke(initial_state(run_id), thread_config(triage_id))
        except Exception as error:
            logger.exception("Triage %s of %s failed", triage_id, run_id)
            self._jobs[triage_id].error = _describe(error)

    async def wait(self, triage_id: str) -> TriageView:
        """The triage once its investigation has finished (or failed)."""
        job = self._jobs.get(triage_id)
        if job is not None:
            await asyncio.wait([job.task])
        return await self.get(triage_id)

    async def get(self, triage_id: str) -> TriageView:
        snapshot = await self._graph.aget_state(thread_config(triage_id))
        values: dict[str, Any] = dict(snapshot.values)
        job = self._jobs.get(triage_id)
        if job is None and not values:
            raise TriageNotFoundError(f"Unknown triage {triage_id!r}.")

        view = TriageView(
            triage_id=triage_id,
            run_id=values.get("run_id") or (job.run_id if job else ""),
            status="failed",
            diagnosis=values.get("diagnosis"),
            decision=values.get("decision"),
        )
        if job is not None and job.error:
            view.error = job.error
        elif job is not None and not job.task.done():
            view.status = "investigating"
        elif view.decision is not None:
            view.status = "approved" if view.decision.approved else "rejected"
        elif snapshot.next == ("approve",):
            view.status = "awaiting_approval"
        else:
            view.error = "The investigation stopped before a diagnosis (did the service restart?)."
        return view

    async def decide(self, triage_id: str, decision: DecisionIn) -> TriageView:
        # The lock stops two concurrent decisions from both resuming the same pause.
        async with self._decide_lock:
            view = await self.get(triage_id)
            if view.status != "awaiting_approval":
                raise NotAwaitingApprovalError(f"Triage {triage_id} is {view.status}.")
            resume: Command[Any] = Command(resume=decision.model_dump())
            await self._graph.ainvoke(resume, thread_config(triage_id))
        return await self.get(triage_id)

    async def aclose(self) -> None:
        running = [job.task for job in self._jobs.values() if not job.task.done()]
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)

"""Typed records for pipeline runs, task logs and table row counts.

These models are both the store's internal representation and the structured
output of the MCP tools, so an MCP client gets a JSON schema for every result.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class RunState(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    RUNNING = "running"


class TaskState(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    UPSTREAM_FAILED = "upstream_failed"
    SKIPPED = "skipped"
    RUNNING = "running"


class TaskInstance(BaseModel):
    task_id: str
    state: TaskState
    try_number: int = Field(ge=1)
    start_date: datetime | None = None
    end_date: datetime | None = None
    log_file: str | None = Field(
        default=None, description="Log path relative to the data directory.", exclude=True
    )


class PipelineRun(BaseModel):
    run_id: str
    dag_id: str
    state: RunState
    logical_date: datetime
    start_date: datetime
    end_date: datetime | None = None
    tasks: list[TaskInstance]

    @property
    def failed_tasks(self) -> list[TaskInstance]:
        return [t for t in self.tasks if t.state is TaskState.FAILED]


class RunSummary(BaseModel):
    """One line per run, as returned by `get_failed_runs`."""

    run_id: str
    dag_id: str
    state: RunState
    logical_date: datetime
    end_date: datetime | None
    failed_tasks: list[str]

    @classmethod
    def from_run(cls, run: PipelineRun) -> RunSummary:
        return cls(
            run_id=run.run_id,
            dag_id=run.dag_id,
            state=run.state,
            logical_date=run.logical_date,
            end_date=run.end_date,
            failed_tasks=[t.task_id for t in run.failed_tasks],
        )


class TaskLog(BaseModel):
    run_id: str
    task_id: str
    try_number: int
    lines: list[str]
    truncated: bool = Field(description="True when only the last `tail_lines` lines are shown.")


class DailyRowCount(BaseModel):
    day: date
    rows: int = Field(ge=0)


class RowCountReport(BaseModel):
    table: str
    counts: list[DailyRowCount] = Field(description="Oldest first.")
    latest_rows: int
    baseline_avg_rows: float = Field(description="Mean of the days before the latest one.")
    pct_change_vs_baseline: float = Field(description="Latest day vs baseline, in percent.")

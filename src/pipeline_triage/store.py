"""Read-only access to pipeline runs, task logs and row counts.

Stage 1 reads a directory of JSON fixtures and plain-text logs, laid out like this:

    <data_dir>/runs.json         list of PipelineRun records
    <data_dir>/row_counts.json   {"<table>": [{"day": ..., "rows": ...}, ...]}
    <data_dir>/logs/...          task logs, referenced by each task's `log_file`

The MCP server only talks to `RunStore`, so a later stage can swap in a
PostgreSQL-backed store without touching the tools.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import TypeAdapter

from pipeline_triage.models import (
    DailyRowCount,
    PipelineRun,
    RowCountReport,
    RunState,
    TaskInstance,
    TaskLog,
)

SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"

_runs_adapter = TypeAdapter(list[PipelineRun])
_counts_adapter = TypeAdapter(dict[str, list[DailyRowCount]])


class NotFoundError(LookupError):
    """Raised when a run, task or table does not exist. The message is safe to show a client."""


class RunStore:
    def __init__(
        self,
        data_dir: Path,
        runs: list[PipelineRun],
        row_counts: dict[str, list[DailyRowCount]],
    ) -> None:
        self._data_dir = data_dir.resolve()
        self._runs = {run.run_id: run for run in runs}
        self._row_counts = {
            table: sorted(c, key=lambda r: r.day) for table, c in row_counts.items()
        }

    @classmethod
    def from_dir(cls, data_dir: Path = SAMPLE_DATA_DIR) -> RunStore:
        runs = _runs_adapter.validate_json((data_dir / "runs.json").read_bytes())
        row_counts = _counts_adapter.validate_json((data_dir / "row_counts.json").read_bytes())
        return cls(data_dir, runs, row_counts)

    def failed_runs(
        self, dag_id: str | None = None, since: datetime | None = None, limit: int = 20
    ) -> list[PipelineRun]:
        """Failed runs, newest first. A `since` without a timezone is read as UTC."""
        if since is not None and since.tzinfo is None:
            since = since.replace(tzinfo=UTC)
        runs = [
            r
            for r in self._runs.values()
            if r.state is RunState.FAILED
            and (dag_id is None or r.dag_id == dag_id)
            and (since is None or r.logical_date >= since)
        ]
        return sorted(runs, key=lambda r: r.logical_date, reverse=True)[:limit]

    def get_run(self, run_id: str) -> PipelineRun:
        try:
            return self._runs[run_id]
        except KeyError:
            raise NotFoundError(f"Unknown run_id {run_id!r}.") from None

    def get_task_log(self, run_id: str, task_id: str, tail_lines: int = 200) -> TaskLog:
        task = self._get_task(self.get_run(run_id), task_id)
        if task.log_file is None:
            raise NotFoundError(f"Task {task_id!r} in run {run_id!r} has no stored log.")
        lines = self._read_log(task.log_file).splitlines()
        return TaskLog(
            run_id=run_id,
            task_id=task_id,
            try_number=task.try_number,
            lines=lines[-tail_lines:],
            truncated=len(lines) > tail_lines,
        )

    def tables(self) -> list[str]:
        return sorted(self._row_counts)

    def row_counts(self, table: str, days: int = 7) -> RowCountReport:
        if table not in self._row_counts:
            raise NotFoundError(
                f"Unknown table {table!r}. Known tables: {', '.join(self.tables())}."
            )
        counts = self._row_counts[table][-days:]
        latest, baseline = counts[-1], counts[:-1]
        baseline_avg = sum(c.rows for c in baseline) / len(baseline) if baseline else latest.rows
        pct_change = (latest.rows - baseline_avg) / baseline_avg * 100 if baseline_avg else 0.0
        return RowCountReport(
            table=table,
            counts=counts,
            latest_rows=latest.rows,
            baseline_avg_rows=round(baseline_avg, 1),
            pct_change_vs_baseline=round(pct_change, 1),
        )

    @staticmethod
    def _get_task(run: PipelineRun, task_id: str) -> TaskInstance:
        for task in run.tasks:
            if task.task_id == task_id:
                return task
        known = ", ".join(t.task_id for t in run.tasks)
        raise NotFoundError(f"Run {run.run_id!r} has no task {task_id!r}. Tasks: {known}.")

    def _read_log(self, relative_path: str) -> str:
        # Log paths come from runs.json, never from the client, but still refuse
        # anything that escapes the data directory.
        path = (self._data_dir / relative_path).resolve()
        if not path.is_relative_to(self._data_dir):
            raise NotFoundError("Log path is outside the data directory.")
        if not path.is_file():
            raise NotFoundError(f"Log file {relative_path!r} is missing.")
        return path.read_text(encoding="utf-8")

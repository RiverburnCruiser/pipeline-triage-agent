"""MCP server that exposes pipeline run status, task logs and row counts as tools.

Run it over stdio (what Claude Code, Claude Desktop and the MCP Inspector expect):

    python -m pipeline_triage.mcp_server

Set TRIAGE_DATA_DIR to point at your own data directory; it defaults to the
bundled sample data. Set DATABASE_URL to also expose `search_runbooks`, which
needs the runbook index built by `pipeline-triage-ingest`.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from pipeline_triage.models import PipelineRun, RowCountReport, RunSummary, TaskLog
from pipeline_triage.runbooks import RunbookHit, RunbookIndex
from pipeline_triage.store import SAMPLE_DATA_DIR, NotFoundError, RunStore

INSTRUCTIONS = """\
Tools for investigating failed data pipeline runs (Airflow-style DAGs).
Start with get_failed_runs, then get_run_status for the task breakdown, then
get_task_log on the failed task. Use query_row_counts to check whether a table's
volume dropped, which points to partial loads or upstream data issues. When
search_runbooks is available, search it with the error lines from the log to find
the matching runbook and its fix."""

READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False)


def build_server(store: RunStore, runbooks: RunbookIndex | None = None) -> MCPServer:
    server = MCPServer("pipeline-triage", instructions=INSTRUCTIONS, version="0.1.0")

    @server.tool(annotations=READ_ONLY)
    def get_failed_runs(
        dag_id: Annotated[str | None, Field(description="Only runs of this DAG.")] = None,
        since: Annotated[
            datetime | None, Field(description="Only runs with logical_date at or after this.")
        ] = None,
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
    ) -> list[RunSummary]:
        """List failed pipeline runs, newest first, with the tasks that failed in each."""
        return [RunSummary.from_run(r) for r in store.failed_runs(dag_id, since, limit)]

    @server.tool(annotations=READ_ONLY)
    def get_run_status(run_id: str) -> PipelineRun:
        """Get one run's state and every task's state, try number and timings."""
        try:
            return store.get_run(run_id)
        except NotFoundError as e:
            raise ToolError(str(e)) from e

    @server.tool(annotations=READ_ONLY)
    def get_task_log(
        run_id: str,
        task_id: str,
        tail_lines: Annotated[int, Field(ge=1, le=2000)] = 200,
    ) -> TaskLog:
        """Get the last `tail_lines` lines of a task's log for its latest try."""
        try:
            return store.get_task_log(run_id, task_id, tail_lines)
        except NotFoundError as e:
            raise ToolError(str(e)) from e

    @server.tool(annotations=READ_ONLY)
    def query_row_counts(
        table: str,
        days: Annotated[int, Field(ge=2, le=90)] = 7,
    ) -> RowCountReport:
        """Get daily row counts for a warehouse table and compare the latest day to the
        average of the days before it. A large negative change suggests a partial load."""
        try:
            return store.row_counts(table, days)
        except NotFoundError as e:
            raise ToolError(str(e)) from e

    @server.tool(annotations=READ_ONLY)
    def list_tables() -> list[str]:
        """List the warehouse tables that query_row_counts can inspect."""
        return store.tables()

    if runbooks is not None:
        index = runbooks

        @server.tool(annotations=READ_ONLY)
        def search_runbooks(
            query: Annotated[
                str, Field(description="Error message or symptoms, e.g. lines from a task log.")
            ],
            k: Annotated[int, Field(ge=1, le=10)] = 3,
        ) -> list[RunbookHit]:
            """Find the runbooks that best match an error, ranked by semantic similarity.
            Each hit includes the full runbook with its diagnosis and fix steps."""
            return index.search(query, k)

    return server


def main() -> None:
    data_dir = Path(os.environ.get("TRIAGE_DATA_DIR", SAMPLE_DATA_DIR))
    runbooks = None
    if database_url := os.environ.get("DATABASE_URL"):
        from pipeline_triage.embeddings import FastEmbedEmbedder

        runbooks = RunbookIndex.connect(database_url, FastEmbedEmbedder())
    build_server(RunStore.from_dir(data_dir), runbooks).run()


if __name__ == "__main__":
    main()

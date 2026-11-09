"""End-to-end tests: a real MCP client talking to the server in-process."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from mcp import Client

from pipeline_triage.mcp_server import build_server
from pipeline_triage.runbooks import RunbookIndex
from pipeline_triage.store import RunStore

# anyio's pytest plugin (installed with mcp) runs each test and its async fixtures
# in one task, which the MCP client's task group requires.
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client(store: RunStore) -> AsyncIterator[Client]:
    async with Client(build_server(store)) as c:
        yield c


async def call(client: Client, tool: str, **arguments: Any) -> Any:
    result = await client.call_tool(tool, arguments)
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return result.structured_content


async def test_lists_all_tools_as_read_only(client: Client) -> None:
    tools = (await client.list_tools()).tools
    assert {t.name for t in tools} == {
        "get_failed_runs",
        "get_run_status",
        "get_task_log",
        "query_row_counts",
        "list_tables",
    }
    for tool in tools:
        assert tool.description
        assert tool.output_schema
        assert tool.annotations and tool.annotations.read_only_hint


async def test_triage_walkthrough(client: Client) -> None:
    """The path an agent would take: failed runs -> run status -> failing task's log."""
    failed = (await call(client, "get_failed_runs", dag_id="vendor_payments_sync"))["result"]
    assert [r["run_id"] for r in failed] == ["vendor_payments_sync__2026-10-29"]
    assert failed[0]["failed_tasks"] == ["extract_vendor_api"]

    run = await call(client, "get_run_status", run_id=failed[0]["run_id"])
    states = {t["task_id"]: t["state"] for t in run["tasks"]}
    assert states == {
        "extract_vendor_api": "failed",
        "transform_payments": "upstream_failed",
        "load_warehouse": "upstream_failed",
    }
    assert all("log_file" not in t for t in run["tasks"])  # internal path stays private

    log = await call(client, "get_task_log", run_id=run["run_id"], task_id="extract_vendor_api")
    assert log["try_number"] == 2
    assert any("Client secret has expired" in line for line in log["lines"])


async def test_query_row_counts(client: Client) -> None:
    report = await call(client, "query_row_counts", table="analytics.trip_events", days=7)
    assert report["pct_change_vs_baseline"] == pytest.approx(-62.2)
    tables = (await call(client, "list_tables"))["result"]
    assert "analytics.trip_events" in tables


@pytest.mark.parametrize(
    ("tool", "arguments", "message"),
    [
        ("get_run_status", {"run_id": "nope"}, "Unknown run_id"),
        ("get_task_log", {"run_id": "trip_events_daily__2026-10-28", "task_id": "x"}, "no task"),
        ("query_row_counts", {"table": "analytics.nope"}, "Known tables"),
        ("get_failed_runs", {"limit": 0}, "limit"),
    ],
)
async def test_bad_input_returns_tool_error(
    client: Client, tool: str, arguments: dict[str, Any], message: str
) -> None:
    result = await client.call_tool(tool, arguments)
    assert result.is_error
    assert message in str(result.content)


@pytest.mark.db
async def test_search_runbooks_is_exposed_when_an_index_is_given(
    store: RunStore, hashing_index: RunbookIndex
) -> None:
    async with Client(build_server(store, hashing_index)) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert "search_runbooks" in names
        hits = (await call(client, "search_runbooks", query="pod OOMKilled SIGKILL", k=2))["result"]
        assert hits[0]["runbook_id"] == "worker-oom"
        assert "## Fix" in hits[0]["body"]

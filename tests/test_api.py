"""The HTTP API, with the scripted model and the real MCP server in process."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from mcp import Client
from test_agent import DIAGNOSIS, RUN_ID, ScriptedModel

from pipeline_triage.api import create_app
from pipeline_triage.diagnosis import Diagnosis
from pipeline_triage.llm import Message, TriageModel
from pipeline_triage.mcp_server import build_server
from pipeline_triage.store import RunStore

pytestmark = pytest.mark.anyio

CALLS = [
    ("get_run_status", {"run_id": RUN_ID}),
    ("get_task_log", {"run_id": RUN_ID, "task_id": "extract_vendor_api"}),
]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def in_process(store: RunStore) -> Callable[[], AbstractAsyncContextManager[Client]]:
    return lambda: Client(build_server(store))


@asynccontextmanager
async def serve(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


def app_with(
    store: RunStore, model: TriageModel | None = None, database_url: str | None = None
) -> FastAPI:
    return create_app(
        model=model or ScriptedModel(CALLS),
        mcp_client=in_process(store),
        database_url=database_url,
    )


@pytest.fixture
async def http(store: RunStore) -> AsyncIterator[httpx.AsyncClient]:
    async with serve(app_with(store)) as http:
        yield http


async def start_and_wait(http: httpx.AsyncClient, run_id: str = RUN_ID) -> dict[str, Any]:
    response = await http.post("/triages", params={"wait": "true"}, json={"run_id": run_id})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


async def test_health_and_failed_runs(http: httpx.AsyncClient) -> None:
    health = (await http.get("/health")).json()
    assert health == {"status": "ok", "runbook_search": False}

    runs = (await http.get("/runs/failed", params={"limit": 2})).json()
    assert len(runs) == 2
    assert {"run_id", "dag_id", "failed_tasks"} <= runs[0].keys()


async def test_triage_pauses_for_approval_then_records_it(http: httpx.AsyncClient) -> None:
    triage = await start_and_wait(http)
    assert triage["status"] == "awaiting_approval"
    assert triage["run_id"] == RUN_ID
    assert Diagnosis.model_validate(triage["diagnosis"]) == DIAGNOSIS

    url = f"/triages/{triage['triage_id']}/decision"
    decided = await http.post(url, json={"approved": True, "note": "rotating now"})
    assert decided.status_code == 200
    assert decided.json()["status"] == "approved"
    assert decided.json()["decision"] == {"approved": True, "note": "rotating now"}

    again = await http.post(url, json={"approved": False})
    assert again.status_code == 409
    fetched = (await http.get(f"/triages/{triage['triage_id']}")).json()
    assert fetched["status"] == "approved"


async def test_rejection(http: httpx.AsyncClient) -> None:
    triage = await start_and_wait(http)
    url = f"/triages/{triage['triage_id']}/decision"
    decided = (await http.post(url, json={"approved": False})).json()
    assert decided["status"] == "rejected"
    assert decided["decision"] == {"approved": False, "note": ""}


async def test_background_start_defaults_to_latest_failure(
    http: httpx.AsyncClient, store: RunStore
) -> None:
    response = await http.post("/triages")
    assert response.status_code == 202
    started = response.json()
    assert started["status"] == "investigating"
    assert started["run_id"] == store.failed_runs(limit=1)[0].run_id
    assert response.headers["location"] == f"/triages/{started['triage_id']}"


async def test_unknown_ids_are_404(http: httpx.AsyncClient) -> None:
    assert (await http.post("/triages", json={"run_id": "no_such_run"})).status_code == 404
    assert (await http.get("/triages/nope")).status_code == 404
    decision = await http.post("/triages/nope/decision", json={"approved": True})
    assert decision.status_code == 404


class BrokenModel(ScriptedModel):
    async def diagnose(self, system: str, messages: list[Message]) -> Diagnosis:
        raise RuntimeError("model went away")


async def test_failed_triage_reports_the_error(store: RunStore) -> None:
    async with serve(app_with(store, BrokenModel(CALLS))) as http:
        triage = await start_and_wait(http)
        assert triage["status"] == "failed"
        assert triage["error"] == "RuntimeError: model went away"
        url = f"/triages/{triage['triage_id']}/decision"
        assert (await http.post(url, json={"approved": True})).status_code == 409


@pytest.mark.db
async def test_pending_approval_survives_a_restart(store: RunStore, database_url: str) -> None:
    async with serve(app_with(store, database_url=database_url)) as http:
        triage = await start_and_wait(http)
    assert triage["status"] == "awaiting_approval"

    # A new app: nothing in memory, only what PostgreSQL checkpointed.
    async with serve(app_with(store, database_url=database_url)) as http:
        url = f"/triages/{triage['triage_id']}"
        assert (await http.get(url)).json()["status"] == "awaiting_approval"
        decided = await http.post(f"{url}/decision", json={"approved": True})
        assert decided.json()["status"] == "approved"

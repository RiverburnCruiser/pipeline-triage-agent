"""HTTP API for the triage agent.

    GET  /health                     liveness, and whether runbook search is on
    GET  /runs/failed                failed runs the agent can triage
    POST /triages                    start a triage (add ?wait=true to block until it pauses)
    GET  /triages/{id}               status, diagnosis and decision
    POST /triages/{id}/decision      approve or reject the proposed fix

Run it with `pipeline-triage-api` (or `docker compose up`), then open /docs.

On startup the service launches the MCP server over stdio, as the CLI does. With
DATABASE_URL set, the agent gets runbook search and paused triages are checkpointed in
PostgreSQL, so a triage awaiting approval survives a restart. The model comes from
TRIAGE_PROVIDER and TRIAGE_MODEL.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from mcp import Client
from pydantic import BaseModel

from pipeline_triage.llm import TriageModel, triage_model_from_env
from pipeline_triage.service import (
    DecisionIn,
    NotAwaitingApprovalError,
    RunNotFoundError,
    TriageNotFoundError,
    TriageService,
    TriageView,
)
from pipeline_triage.toolbox import McpToolbox, stdio_server


class TriageRequest(BaseModel):
    run_id: str | None = None
    """The run to triage. Leave it out to triage the most recent failure."""


class Health(BaseModel):
    status: str
    runbook_search: bool


def _get_service(request: Request) -> TriageService:
    service: TriageService = request.app.state.service
    return service


# Module level, because FastAPI resolves these annotations in the module's namespace.
Service = Annotated[TriageService, Depends(_get_service)]


@asynccontextmanager
async def _checkpointer(database_url: str | None) -> AsyncIterator[BaseCheckpointSaver[Any]]:
    if not database_url:
        yield InMemorySaver()
        return
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(database_url) as saver:
        await saver.setup()
        yield saver


def create_app(
    *,
    model: TriageModel | None = None,
    mcp_client: Callable[[], AbstractAsyncContextManager[Client]] = lambda: Client(stdio_server()),
    database_url: str | None = None,
) -> FastAPI:
    """The app. Defaults to the model from the environment and the stdio MCP server.

    `database_url` is where paused triages are checkpointed; None keeps them in memory.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            client = await stack.enter_async_context(mcp_client())
            toolbox = await McpToolbox(client).load()
            checkpointer = await stack.enter_async_context(_checkpointer(database_url))
            service = TriageService(model or triage_model_from_env(), toolbox, checkpointer)
            stack.push_async_callback(service.aclose)
            app.state.service = service
            yield

    app = FastAPI(
        title="Pipeline Triage Agent",
        summary="Triages failed data pipeline runs and proposes a fix for a human to approve.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health(service: Service) -> Health:
        return Health(
            status="ok",
            runbook_search="search_runbooks" in service.toolbox.names,
        )

    @app.get("/runs/failed")
    async def failed_runs(
        service: Service, limit: Annotated[int, Query(ge=1, le=100)] = 20
    ) -> list[dict[str, Any]]:
        return await service.toolbox.failed_runs(limit)

    @app.post("/triages", status_code=202)
    async def start_triage(
        service: Service,
        response: Response,
        body: TriageRequest | None = None,
        wait: bool = False,
    ) -> TriageView:
        """Start a triage. It runs in the background: poll GET /triages/{id}.

        With `wait=true` the call returns once the diagnosis is ready (or the triage
        failed), which takes from seconds to a few minutes depending on the model.
        """
        try:
            view = await service.start((body or TriageRequest()).run_id)
        except RunNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        response.headers["Location"] = f"/triages/{view.triage_id}"
        if wait:
            view = await service.wait(view.triage_id)
            response.status_code = 200
        return view

    @app.get("/triages/{triage_id}")
    async def get_triage(service: Service, triage_id: str) -> TriageView:
        try:
            return await service.get(triage_id)
        except TriageNotFoundError as error:
            raise HTTPException(404, str(error)) from error

    @app.post("/triages/{triage_id}/decision")
    async def decide(service: Service, triage_id: str, decision: DecisionIn) -> TriageView:
        """Approve or reject the proposed fix. The decision is recorded, not executed."""
        try:
            return await service.decide(triage_id, decision)
        except TriageNotFoundError as error:
            raise HTTPException(404, str(error)) from error
        except NotAwaitingApprovalError as error:
            raise HTTPException(409, str(error)) from error

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the triage agent over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app = create_app(database_url=os.environ.get("DATABASE_URL"))
    uvicorn.run(app, host=args.host, port=args.port)

from __future__ import annotations

import os
from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from pipeline_triage.embeddings import HashingEmbedder
from pipeline_triage.runbooks import RunbookIndex, load_runbooks
from pipeline_triage.store import RunStore


@pytest.fixture(scope="session")
def store() -> RunStore:
    return RunStore.from_dir()


def _database_key(url: str) -> tuple[str, str, str]:
    params = conninfo_to_dict(url)
    return (
        str(params.get("host", "localhost")),
        str(params.get("port", "5432")),
        str(params.get("dbname", "")),
    )


def _create_database_if_missing(url: str) -> None:
    dbname = str(conninfo_to_dict(url)["dbname"])
    with psycopg.connect(make_conninfo(url, dbname="postgres"), autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,)).fetchone():
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))


@pytest.fixture(scope="session")
def database_url() -> str:
    """The test database, created on first use.

    Database tests drop and rebuild the runbook tables, so they must never share a
    database with DATABASE_URL, where `pipeline-triage-ingest` keeps the real index.
    """
    url = os.environ.get("TRIAGE_TEST_DATABASE_URL")
    if not url:
        pytest.skip("TRIAGE_TEST_DATABASE_URL is not set (see README: Running the tests)")
    app_url = os.environ.get("DATABASE_URL")
    if app_url and _database_key(url) == _database_key(app_url):
        pytest.fail(
            "TRIAGE_TEST_DATABASE_URL and DATABASE_URL point at the same database. The tests"
            " rebuild the runbook tables and would wipe the real index. Point"
            " TRIAGE_TEST_DATABASE_URL at a separate database such as triage_test"
            " (see .env.example)."
        )
    _create_database_if_missing(url)
    return url


@pytest.fixture
def hashing_index(database_url: str) -> Iterator[RunbookIndex]:
    """A runbook index built with the download-free test embedder."""
    index = RunbookIndex.connect(database_url, HashingEmbedder())
    index.rebuild(load_runbooks())
    yield index
    index.close()

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from pipeline_triage.embeddings import HashingEmbedder
from pipeline_triage.runbooks import RunbookIndex, load_runbooks
from pipeline_triage.store import RunStore


@pytest.fixture(scope="session")
def store() -> RunStore:
    return RunStore.from_dir()


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.environ.get("TRIAGE_TEST_DATABASE_URL")
    if not url:
        pytest.skip("TRIAGE_TEST_DATABASE_URL is not set (see README: Running the tests)")
    return url


@pytest.fixture
def hashing_index(database_url: str) -> Iterator[RunbookIndex]:
    """A runbook index built with the download-free test embedder."""
    index = RunbookIndex.connect(database_url, HashingEmbedder())
    index.rebuild(load_runbooks())
    yield index
    index.close()

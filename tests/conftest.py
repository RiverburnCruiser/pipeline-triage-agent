from __future__ import annotations

import pytest

from pipeline_triage.store import RunStore


@pytest.fixture(scope="session")
def store() -> RunStore:
    return RunStore.from_dir()

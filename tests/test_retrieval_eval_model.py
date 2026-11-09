"""Retrieval quality with the real embedding model. Runs in CI.

Locally: docker compose up -d, then
    TRIAGE_TEST_DATABASE_URL=... TRIAGE_TEST_REAL_MODEL=1 pytest -m model
"""

from __future__ import annotations

import os

import pytest

from pipeline_triage.embeddings import FastEmbedEmbedder
from pipeline_triage.retrieval_eval import evaluate, format_report, load_labels
from pipeline_triage.runbooks import RunbookIndex, load_runbooks
from pipeline_triage.store import RunStore

pytestmark = [pytest.mark.db, pytest.mark.model]


def test_real_model_ranks_the_right_runbook_first(database_url: str, store: RunStore) -> None:
    if os.environ.get("TRIAGE_TEST_REAL_MODEL") != "1":
        pytest.skip("set TRIAGE_TEST_REAL_MODEL=1 to download and test the real model")
    index = RunbookIndex.connect(database_url, FastEmbedEmbedder())
    try:
        index.rebuild(load_runbooks())
        report = evaluate(index, store, load_labels(), k=3)
    finally:
        index.close()
    print(format_report(report))
    assert report.hit_rate(1) == 1.0, format_report(report)

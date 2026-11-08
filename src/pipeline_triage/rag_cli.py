"""Command-line entry points for the runbook index.

    pipeline-triage-ingest           load runbooks into PostgreSQL (pgvector)
    pipeline-triage-eval-retrieval   score search against the labelled incidents

Both read DATABASE_URL. TRIAGE_RUNBOOKS_DIR overrides the bundled runbooks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pipeline_triage.embeddings import FastEmbedEmbedder
from pipeline_triage.retrieval_eval import evaluate, format_report, load_labels
from pipeline_triage.runbooks import SAMPLE_RUNBOOKS_DIR, RunbookIndex, load_runbooks
from pipeline_triage.store import RunStore


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL is not set. See .env.example.")
    return url


def ingest() -> None:
    runbooks = load_runbooks(Path(os.environ.get("TRIAGE_RUNBOOKS_DIR", SAMPLE_RUNBOOKS_DIR)))
    index = RunbookIndex.connect(_database_url(), FastEmbedEmbedder())
    try:
        chunks = index.rebuild(runbooks)
    finally:
        index.close()
    print(f"Indexed {len(runbooks)} runbooks as {chunks} sections.")


def eval_retrieval() -> None:
    index = RunbookIndex.connect(_database_url(), FastEmbedEmbedder())
    try:
        print(format_report(evaluate(index, RunStore.from_dir(), load_labels())))
    finally:
        index.close()

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline_triage.embeddings import HashingEmbedder
from pipeline_triage.retrieval_eval import evaluate, load_labels, log_evidence
from pipeline_triage.runbooks import Runbook, RunbookIndex, load_runbooks, split_sections
from pipeline_triage.store import RunStore


def test_load_runbooks_reads_every_file_with_a_title() -> None:
    runbooks = load_runbooks()
    assert len(runbooks) == 9
    by_id = {rb.id: rb for rb in runbooks}
    assert by_id["worker-oom"].title == "Task killed for running out of memory"


def test_load_runbooks_rejects_a_file_without_a_title(tmp_path: Path) -> None:
    (tmp_path / "bad.md").write_text("no heading here\n")
    with pytest.raises(ValueError, match="bad.md"):
        load_runbooks(tmp_path)


def test_split_sections() -> None:
    runbook = Runbook(
        id="x",
        title="X",
        body="# X\nintro is ignored\n## Symptoms\nboom\n## Empty\n\n## Fix\n- restart\n- rerun",
    )
    chunks = split_sections(runbook)
    assert [(c.section, c.content) for c in chunks] == [
        ("Symptoms", "boom"),
        ("Fix", "- restart\n- rerun"),
    ]


def test_every_label_points_at_a_real_runbook_and_failed_task(store: RunStore) -> None:
    runbook_ids = {rb.id for rb in load_runbooks()}
    for run_id, label in load_labels().items():
        assert label.runbook_id in runbook_ids
        assert label.failed_task in [t.task_id for t in store.get_run(run_id).failed_tasks]


def test_log_evidence_keeps_pipeline_messages_and_drops_airflow_noise() -> None:
    lines = [
        "[t] {taskinstance.py:1359} INFO - Starting attempt 1 of 1",
        "[t] {validate.py:48} WARNING - Unexpected column in source: speed_kph",
        "[t] {reconcile.py:35} INFO - Manifest lists 96 objects; 37 present",
        "[t] {taskinstance.py:2698} ERROR - Task failed with exception",
        "Traceback (most recent call last):",
        '  File "x.py", line 1, in f',
        "trip_events.errors.SchemaContractError: schema mismatch",
        "[t] {local_task_job_runner.py:228} INFO - Task exited with return code 1",
    ]
    assert log_evidence(lines) == "\n".join([lines[1], lines[2], lines[6]])


@pytest.mark.db
def test_rebuild_is_repeatable(hashing_index: RunbookIndex) -> None:
    runbooks = load_runbooks()
    expected_chunks = sum(len(split_sections(rb)) for rb in runbooks)
    assert hashing_index.rebuild(runbooks) == expected_chunks
    assert hashing_index.rebuild(runbooks) == expected_chunks


@pytest.mark.db
def test_search_returns_distinct_full_runbooks(hashing_index: RunbookIndex) -> None:
    hits = hashing_index.search("HTTP 401 Unauthorized invalid_client client secret expired", k=3)
    assert hits[0].runbook_id == "expired-credentials"
    assert hits[0].body.startswith("# Expired or revoked API credentials")
    assert len({h.runbook_id for h in hits}) == len(hits) == 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


@pytest.mark.db
def test_rebuild_handles_a_new_embedding_size(database_url: str) -> None:
    for dim in (64, 128):
        index = RunbookIndex.connect(database_url, HashingEmbedder(dim=dim))
        index.rebuild(load_runbooks())
        assert index.search("No space left on device", k=1)[0].runbook_id == "disk-full-staging"
        index.close()


@pytest.mark.db
def test_keyword_embedder_finds_every_incident_in_top_3(
    hashing_index: RunbookIndex, store: RunStore
) -> None:
    """A floor for the pipeline itself; the real model is held to a higher bar."""
    report = evaluate(hashing_index, store, load_labels(), k=3)
    assert report.hit_rate(3) == 1.0, report.results

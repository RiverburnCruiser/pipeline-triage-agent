from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from pipeline_triage.models import RunState, TaskState
from pipeline_triage.store import SAMPLE_DATA_DIR, NotFoundError, RunStore


def test_every_referenced_log_file_exists(store: RunStore) -> None:
    for run in store.failed_runs(limit=100):
        for task in run.tasks:
            if task.log_file:
                assert (SAMPLE_DATA_DIR / task.log_file).is_file(), task.log_file


def test_every_failed_run_has_a_failed_task_with_a_log(store: RunStore) -> None:
    for run in store.failed_runs(limit=100):
        assert run.failed_tasks, run.run_id
        assert all(t.log_file for t in run.failed_tasks), run.run_id


def test_failed_runs_are_newest_first_and_only_failed(store: RunStore) -> None:
    runs = store.failed_runs()
    assert len(runs) == 5
    assert all(r.state is RunState.FAILED for r in runs)
    dates = [r.logical_date for r in runs]
    assert dates == sorted(dates, reverse=True)


def test_failed_runs_filters(store: RunStore) -> None:
    assert [r.run_id for r in store.failed_runs(dag_id="trip_events_daily")] == [
        "trip_events_daily__2026-10-30",
        "trip_events_daily__2026-10-28",
    ]
    # A naive `since` is treated as UTC rather than raising on comparison.
    recent = store.failed_runs(since=datetime(2026, 10, 30))
    assert {r.run_id for r in recent} == {
        "trip_events_daily__2026-10-30",
        "device_health_rollup__2026-10-30",
    }
    assert len(store.failed_runs(limit=2)) == 2


def test_get_run_unknown_id(store: RunStore) -> None:
    with pytest.raises(NotFoundError, match="Unknown run_id"):
        store.get_run("nope")


def test_get_task_log_returns_tail(store: RunStore) -> None:
    log = store.get_task_log("trip_events_daily__2026-10-28", "validate_schema")
    assert not log.truncated
    assert any("SchemaContractError" in line for line in log.lines)

    tail = store.get_task_log("trip_events_daily__2026-10-28", "validate_schema", tail_lines=2)
    assert tail.truncated
    assert tail.lines == log.lines[-2:]


def test_get_task_log_errors(store: RunStore) -> None:
    run_id = "trip_events_daily__2026-10-28"
    with pytest.raises(NotFoundError, match="has no task"):
        store.get_task_log(run_id, "missing_task")
    task = next(t for t in store.get_run(run_id).tasks if t.task_id == "load_warehouse")
    assert task.state is TaskState.UPSTREAM_FAILED
    with pytest.raises(NotFoundError, match="no stored log"):
        store.get_task_log(run_id, "load_warehouse")


def test_log_path_cannot_escape_data_dir(tmp_path: Path) -> None:
    (tmp_path / "runs.json").write_text(
        """[{"run_id": "r1", "dag_id": "d", "state": "failed",
             "logical_date": "2026-10-01T00:00:00Z", "start_date": "2026-10-01T00:00:00Z",
             "tasks": [{"task_id": "t", "state": "failed", "try_number": 1,
                        "log_file": "../../etc/passwd"}]}]"""
    )
    (tmp_path / "row_counts.json").write_text("{}")
    with pytest.raises(NotFoundError, match="outside the data directory"):
        RunStore.from_dir(tmp_path).get_task_log("r1", "t")


def test_row_counts_flags_the_trip_events_drop(store: RunStore) -> None:
    report = store.row_counts("analytics.trip_events")
    assert len(report.counts) == 7
    assert report.latest_rows == 1_402_377
    assert report.baseline_avg_rows == pytest.approx(3_711_840.5)
    assert report.pct_change_vs_baseline == pytest.approx(-62.2)


def test_row_counts_respects_days_and_unknown_table(store: RunStore) -> None:
    assert len(store.row_counts("analytics.trip_events", days=3).counts) == 3
    with pytest.raises(NotFoundError, match="Known tables"):
        store.row_counts("analytics.nope")

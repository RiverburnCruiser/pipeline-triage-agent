"""Score runbook search against the labelled sample incidents.

For each labelled failed run, the query is the evidence from the failing task's
log (see `log_evidence`), which is what an agent would search with. A hit
means the expected runbook came back in the top k.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from pipeline_triage.runbooks import RunbookIndex
from pipeline_triage.store import SAMPLE_DATA_DIR, RunStore


class IncidentLabel(BaseModel):
    failed_task: str
    root_cause: str
    runbook_id: str


class IncidentResult(BaseModel):
    run_id: str
    expected: str
    retrieved: list[str]

    @property
    def rank(self) -> int | None:
        """1-based rank of the expected runbook, or None if it was not retrieved."""
        return self.retrieved.index(self.expected) + 1 if self.expected in self.retrieved else None


class RetrievalReport(BaseModel):
    k: int
    results: list[IncidentResult]

    def hit_rate(self, at: int) -> float:
        hits = sum(1 for r in self.results if r.rank is not None and r.rank <= at)
        return hits / len(self.results)

    @property
    def mrr(self) -> float:
        return sum(1 / r.rank for r in self.results if r.rank) / len(self.results)


def load_labels(path: Path = SAMPLE_DATA_DIR / "labels.json") -> dict[str, IncidentLabel]:
    return TypeAdapter(dict[str, IncidentLabel]).validate_json(path.read_bytes())


_FRAMEWORK_SOURCES = ("{taskinstance.py:", "{local_task_job_runner.py:")


def log_evidence(lines: list[str]) -> str:
    """The log minus Airflow's own bookkeeping lines and traceback frames.

    What is left are the pipeline's own messages and the exception text, which
    carry the diagnosis (for example "37 of 96 objects present").
    """
    keep = [
        line
        for line in lines
        if not any(source in line for source in _FRAMEWORK_SOURCES)
        and not line.startswith(("Traceback", "  "))
    ]
    return "\n".join(keep)


def evaluate(
    index: RunbookIndex, store: RunStore, labels: dict[str, IncidentLabel], k: int = 3
) -> RetrievalReport:
    results = []
    for run_id, label in labels.items():
        log = store.get_task_log(run_id, label.failed_task, tail_lines=500)
        hits = index.search(log_evidence(log.lines), k=k)
        results.append(
            IncidentResult(
                run_id=run_id, expected=label.runbook_id, retrieved=[h.runbook_id for h in hits]
            )
        )
    return RetrievalReport(k=k, results=results)


def format_report(report: RetrievalReport) -> str:
    rows = [f"{'run':40} {'expected':26} rank  top {report.k}"]
    for r in report.results:
        rank = str(r.rank) if r.rank else "-"
        rows.append(f"{r.run_id:40} {r.expected:26} {rank:5} {', '.join(r.retrieved)}")
    rows.append("")
    rows.append(
        f"hit@1 {report.hit_rate(1):.0%}   hit@{report.k} {report.hit_rate(report.k):.0%}"
        f"   MRR {report.mrr:.2f}   ({len(report.results)} incidents)"
    )
    return "\n".join(rows)

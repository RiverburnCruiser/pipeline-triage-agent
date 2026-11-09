"""The structured result of a triage: what the agent concluded and why."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RootCause = Literal[
    "schema_drift",
    "upstream_dependency",
    "credential_expiry",
    "out_of_memory",
    "partial_source_load",
    "rate_limit",
    "duplicate_load",
    "lock_contention",
    "disk_full",
    "unknown",
]


class Diagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    root_cause: RootCause = Field(
        description="The category that best explains the failure. Use 'unknown' when the"
        " evidence does not support one."
    )
    summary: str = Field(description="Two or three sentences an on-call engineer can act on.")
    evidence: list[str] = Field(
        description="Specific facts from the tools that support the root cause, e.g. log lines."
    )
    runbook_id: str | None = Field(
        description="The id of the runbook that matches, or null if none was found."
    )
    fix_steps: list[str] = Field(description="Ordered steps to fix the failure and rerun.")
    related_run_ids: list[str] = Field(
        description="Other failed runs that caused or were caused by this one."
    )
    confidence: Literal["high", "medium", "low"]

    def to_markdown(self) -> str:
        lines = [
            f"## Triage: {self.run_id}",
            "",
            f"**Root cause:** {self.root_cause} ({self.confidence} confidence)",
            "",
            self.summary,
            "",
            "**Evidence**",
            *[f"- {e}" for e in self.evidence],
            "",
            f"**Runbook:** {self.runbook_id or 'none found'}",
            "",
            "**Proposed fix**",
            *[f"{i}. {step}" for i, step in enumerate(self.fix_steps, 1)],
        ]
        if self.related_run_ids:
            lines += ["", f"**Related runs:** {', '.join(self.related_run_ids)}"]
        return "\n".join(lines)

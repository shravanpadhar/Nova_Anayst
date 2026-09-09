"""Append-only audit log for every tool call: input, output size, latency,
and whether a governance policy rule fired. Backs the "Download audit log"
button in the Streamlit UI and gives evals a ground truth for tool-error rate.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AuditEntry:
    call_id: str
    tool_name: str
    input: dict[str, Any]
    started_at: float
    latency_ms: float
    output_summary: str
    output_rows: int | None
    policy_hit: bool
    policy_reason: str | None
    error: str | None


class AuditLog:
    """In-memory, thread-safe audit log for a single agent run. Serializable
    to JSON for the Streamlit download button and for eval scoring.
    """

    def __init__(self):
        self._entries: list[AuditEntry] = []
        self._lock = threading.Lock()

    def record(
        self,
        tool_name: str,
        input: dict[str, Any],
        started_at: float,
        output_summary: str,
        output_rows: int | None = None,
        policy_hit: bool = False,
        policy_reason: str | None = None,
        error: str | None = None,
    ) -> AuditEntry:
        entry = AuditEntry(
            call_id=str(uuid.uuid4())[:8],
            tool_name=tool_name,
            input=input,
            started_at=started_at,
            latency_ms=round((time.time() - started_at) * 1000, 2),
            output_summary=output_summary[:500],
            output_rows=output_rows,
            policy_hit=policy_hit,
            policy_reason=policy_reason,
            error=error,
        )
        with self._lock:
            self._entries.append(entry)
        return entry

    def entries(self) -> list[AuditEntry]:
        with self._lock:
            return list(self._entries)

    def to_json(self) -> str:
        with self._lock:
            return json.dumps([asdict(e) for e in self._entries], indent=2, default=str)

    def summary(self) -> dict[str, Any]:
        entries = self.entries()
        n = len(entries)
        n_errors = sum(1 for e in entries if e.error)
        n_policy_hits = sum(1 for e in entries if e.policy_hit)
        avg_latency = round(sum(e.latency_ms for e in entries) / n, 1) if n else 0.0
        return {
            "total_tool_calls": n,
            "tool_errors": n_errors,
            "policy_hits": n_policy_hits,
            "avg_latency_ms": avg_latency,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

"""Typed state threaded through the LangGraph graph. Kept as a plain
TypedDict (not pydantic) so it works directly with LangGraph's state-graph
reducers and stays trivially JSON-serializable for the audit/report artifacts.
"""
from __future__ import annotations

from typing import Any, TypedDict

from tools.audit import AuditLog
from tools.governance import GovernancePolicy


class ToolCallRecord(TypedDict):
    tool: str
    input: dict[str, Any]
    output: dict[str, Any]
    policy_hit: bool


class StepRecord(TypedDict):
    step_number: int
    step_description: str
    tool_calls: list[ToolCallRecord]
    observation: str


class CriticVerdict(TypedDict):
    sufficient: bool
    issues: list[str]
    next_action: str  # "continue" | "replan" | "finish"
    notes: str


# Hard caps -- never allow unbounded loops/steps/tokens.
MAX_STEPS = 6
MAX_LOOPS = 2
MAX_TOKENS_PER_STEP = 1500


class NovaState(TypedDict, total=False):
    question: str
    plan: list[str]
    step_records: list[StepRecord]
    critic_verdicts: list[CriticVerdict]
    governance_notes: list[str]
    loop_count: int
    status: str  # "planning" | "executing" | "critiquing" | "reporting" | "done" | "insufficient_data"
    final_report: str
    chart_specs: list[dict[str, Any]]
    audit: AuditLog
    events: list[dict[str, Any]]  # append-only event log for UI streaming
    evidence_store: dict[str, Any]  # name -> rows, populated by run_sql/run_python results
    current_step_idx: int

    # Active dataset context -- defaults to the built-in Olist demo when not
    # set explicitly (see agent/graph.py::run_agent). Lets the same graph
    # analyze an arbitrary user-uploaded dataset instead.
    db_path: str
    policy: GovernancePolicy
    dataset_label: str
    is_builtin_olist: bool
    available_tables: list[str]
    semantic_note: str  # built once by node_plan, reused by node_execute

"""Read-only SQL tools over the local DuckDB warehouse. Every query is
validated by sqlglot and checked against tools/governance.py BEFORE it
touches DuckDB. These functions are the ones exposed over MCP in
mcp_server/server.py, and are also unit-testable directly (see tests/).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from tools import governance
from tools.audit import AuditLog

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = ROOT / (os.environ.get("DUCKDB_PATH") or "data/olist.duckdb")

_POLICY = governance.load_policy()


def _connect(db_path: str | Path = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path), read_only=True)


def list_tables(db_path: str | Path = DEFAULT_DB_PATH, audit: AuditLog | None = None) -> dict[str, Any]:
    """List all tables available in the warehouse, with row counts."""
    started = time.time()
    try:
        con = _connect(db_path)
        rows = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name").fetchall()
        tables = []
        for (name,) in rows:
            n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            tables.append({"table": name, "row_count": n})
        con.close()
        result = {"tables": tables}
        if audit:
            audit.record("list_tables", {}, started, f"{len(tables)} tables", output_rows=len(tables))
        return result
    except Exception as e:
        if audit:
            audit.record("list_tables", {}, started, "", error=str(e))
        raise


def get_schema(
    table: str | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    audit: AuditLog | None = None,
    policy: governance.GovernancePolicy | None = None,
) -> dict[str, Any]:
    """Get column names/types for one table (or all tables if table=None).
    Governed columns are annotated with governance="blocked"|"masked" rather
    than silently omitted, so the agent can SEE that a column exists and
    reason about why it can't use it. `policy` defaults to the built-in
    config/governance.yaml policy -- pass a different GovernancePolicy (e.g.
    from tools.governance.build_policy_from_rules) to govern a different
    dataset (see the "upload your own data" flow in app/streamlit_app.py).
    """
    policy = policy or _POLICY
    started = time.time()
    try:
        con = _connect(db_path)
        if table:
            table_list = [table]
        else:
            table_list = [r[0] for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()]

        schema: dict[str, list[dict]] = {}
        for t in table_list:
            cols = con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'main' AND table_name = ? ORDER BY ordinal_position",
                [t],
            ).fetchall()
            schema[t] = [{"column": c, "type": ty} for c, ty in cols]
        con.close()

        annotated = governance.annotate_schema(schema, policy)
        if audit:
            audit.record("get_schema", {"table": table}, started, f"{len(annotated)} table(s)", output_rows=len(annotated))
        return {"schema": annotated}
    except Exception as e:
        if audit:
            audit.record("get_schema", {"table": table}, started, "", error=str(e))
        raise


def run_sql(
    sql: str,
    db_path: str | Path = DEFAULT_DB_PATH,
    audit: AuditLog | None = None,
    max_rows: int | None = None,
    policy: governance.GovernancePolicy | None = None,
) -> dict[str, Any]:
    """Execute a read-only SQL query against the warehouse.

    Returns one of:
      {"rows": [...], "columns": [...], "row_count": N}
      {"policy_violation": true, "reason": "...", "kind": "blocked_column"|"write_denied"|"parse_error"}
      {"error": "..."}

    `policy` defaults to the built-in config/governance.yaml policy -- pass a
    different GovernancePolicy to govern a different dataset.
    """
    policy = policy or _POLICY
    started = time.time()
    max_rows = max_rows or policy.max_rows_returned

    try:
        governance.check_statement_type(sql)
    except governance.PolicyViolation as e:
        result = {"policy_violation": True, "reason": e.reason, "kind": e.kind}
        if audit:
            audit.record("run_sql", {"sql": sql}, started, e.reason, policy_hit=True, policy_reason=e.reason)
        return result

    blocked_hits = governance.find_blocked_column_refs(sql, policy)
    if blocked_hits:
        reasons = [policy.blocked_reason(t, c) for (t, c) in blocked_hits]
        cols_str = ", ".join(f"{t}.{c}" for t, c in blocked_hits)
        reason = f"Query references governed/blocked column(s): {cols_str}. {' '.join(r for r in reasons if r)}"
        result = {"policy_violation": True, "reason": reason, "kind": "blocked_column", "columns": [f"{t}.{c}" for t, c in blocked_hits]}
        if audit:
            audit.record("run_sql", {"sql": sql}, started, reason, policy_hit=True, policy_reason=reason)
        return result

    try:
        con = _connect(db_path)
        df: pd.DataFrame = con.execute(sql).fetch_df()
        con.close()

        truncated = False
        if len(df) > max_rows:
            df = df.head(max_rows)
            truncated = True

        df = governance.mask_dataframe_columns(df, {}, policy)

        result = {
            "columns": list(df.columns),
            "rows": df.to_dict(orient="records"),
            "row_count": len(df),
            "truncated": truncated,
        }
        if audit:
            audit.record("run_sql", {"sql": sql}, started, f"{len(df)} rows returned", output_rows=len(df))
        return result
    except Exception as e:
        result = {"error": f"SQL execution failed: {e}"}
        if audit:
            audit.record("run_sql", {"sql": sql}, started, "", error=str(e))
        return result

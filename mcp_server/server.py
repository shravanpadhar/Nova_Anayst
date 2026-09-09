"""Nova Analyst MCP server -- exposes the governed data tools (tools/) over
the Model Context Protocol so ANY MCP-compatible client (this project's own
LangGraph agent, Claude Desktop, or NuoData's Nora/MCP registry) can call
them identically. This is the "MCP-native" story: the tools aren't bespoke
LangGraph functions, they're a standalone MCP server the agent happens to
consume.

Run standalone (stdio transport):
    python mcp_server/server.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp.server.fastmcp import FastMCP

from tools import analysis_tools, sql_tools
from tools.audit import AuditLog

mcp = FastMCP("nova-analyst")

# A process-wide audit log for calls made directly over MCP (e.g. from
# Claude Desktop). The LangGraph agent in this repo uses its own per-run
# AuditLog instance instead (see agent/graph.py) so each question's audit
# trail is isolated.
_SERVER_AUDIT = AuditLog()


@mcp.tool()
def list_tables() -> str:
    """List all tables in the Nova Analyst data warehouse, with row counts."""
    return json.dumps(sql_tools.list_tables(audit=_SERVER_AUDIT))


@mcp.tool()
def get_schema(table: str = "") -> str:
    """Get column names and types for a table (or all tables if omitted).
    Governed columns are annotated with governance="blocked"|"masked".
    """
    return json.dumps(sql_tools.get_schema(table=table or None, audit=_SERVER_AUDIT))


@mcp.tool()
def run_sql(sql: str) -> str:
    """Execute a read-only SQL SELECT query against the warehouse. Governed
    columns are blocked or masked automatically per config/governance.yaml.
    Never write/DDL SQL -- it will be rejected.
    """
    return json.dumps(sql_tools.run_sql(sql, audit=_SERVER_AUDIT))


@mcp.tool()
def run_python(code: str, dataframes_json: str) -> str:
    """Run pandas analysis code over named result sets. `dataframes_json` is
    a JSON object mapping name -> list of row dicts (typically the "rows"
    field from a prior run_sql call). Code must assign its answer to `result`.
    """
    dataframes = json.loads(dataframes_json) if dataframes_json else {}
    return json.dumps(analysis_tools.run_python(code, dataframes, audit=_SERVER_AUDIT))


@mcp.tool()
def make_chart(rows_json: str, chart_type: str, x: str, y: str = "", color: str = "", title: str = "") -> str:
    """Build a Plotly chart from tabular rows (JSON list of row dicts).
    chart_type: bar|line|scatter|pie|hist.
    """
    rows = json.loads(rows_json)
    result = analysis_tools.make_chart(
        rows, chart_type, x, y=y or None, color=color or None, title=title, audit=_SERVER_AUDIT
    )
    return json.dumps(result)


if __name__ == "__main__":
    mcp.run(transport="stdio")

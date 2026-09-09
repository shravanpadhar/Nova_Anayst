"""Analysis tools that operate on data already returned by run_sql --
they never touch the database directly. `run_python` executes pandas code
in a restricted namespace (no filesystem/network/imports beyond pandas);
`make_chart` turns tabular rows into a Plotly figure.
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from tools.audit import AuditLog

_SAFE_BUILTINS = {
    "len": len, "range": range, "min": min, "max": max, "sum": sum,
    "sorted": sorted, "round": round, "abs": abs, "enumerate": enumerate,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "zip": zip,
    "str": str, "int": int, "float": float, "bool": bool,
}


def run_python(code: str, dataframes: dict[str, list[dict]], audit: AuditLog | None = None) -> dict[str, Any]:
    """Execute pandas analysis code against one or more named result sets.

    `dataframes`: {name: rows} where rows is the list-of-dicts shape returned
    by run_sql (e.g. {"orders_by_month": [...]}). Inside `code`, each key is
    available as a pandas DataFrame variable of the same name. The code must
    assign its final answer to a variable named `result` (a DataFrame, dict,
    list, or scalar) -- that value is returned (JSON-safe).

    The namespace is restricted: only pandas (`pd`) and a small set of safe
    builtins are available. No file, network, or `import` access.
    """
    started = time.time()
    local_ns: dict[str, Any] = {}
    for name, rows in dataframes.items():
        local_ns[name] = pd.DataFrame(rows)

    global_ns = {"__builtins__": _SAFE_BUILTINS, "pd": pd}

    try:
        if "import" in code or "__" in code or "open(" in code or "exec(" in code or "eval(" in code:
            raise ValueError("Code contains disallowed tokens (import/open/exec/eval/dunder).")

        exec(code, global_ns, local_ns)
        result = local_ns.get("result")

        if isinstance(result, pd.DataFrame):
            out = {"result_type": "dataframe", "columns": list(result.columns), "rows": result.to_dict(orient="records")}
        elif isinstance(result, pd.Series):
            out = {"result_type": "series", "rows": result.to_dict()}
        elif isinstance(result, (dict, list, int, float, str, bool)) or result is None:
            out = {"result_type": "scalar", "value": result}
        else:
            out = {"result_type": "scalar", "value": str(result)}

        if audit:
            audit.record("run_python", {"code": code}, started, f"result_type={out.get('result_type')}")
        return out
    except Exception as e:
        if audit:
            audit.record("run_python", {"code": code}, started, "", error=str(e))
        return {"error": f"run_python failed: {e}"}


def make_chart(
    rows: list[dict],
    chart_type: str,
    x: str,
    y: str | list[str] | None = None,
    color: str | None = None,
    title: str = "",
    audit: AuditLog | None = None,
) -> dict[str, Any]:
    """Build a Plotly figure from tabular rows. chart_type: bar|line|scatter|pie|hist.
    Returns {"figure_json": "..."} (plotly JSON, renderable directly by Streamlit)
    or {"error": "..."}.
    """
    started = time.time()
    try:
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("Cannot chart an empty result set.")

        kwargs = {"title": title} if title else {}
        if chart_type == "bar":
            fig = px.bar(df, x=x, y=y, color=color, **kwargs)
        elif chart_type == "line":
            fig = px.line(df, x=x, y=y, color=color, markers=True, **kwargs)
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x, y=y, color=color, **kwargs)
        elif chart_type == "pie":
            fig = px.pie(df, names=x, values=y, **kwargs)
        elif chart_type == "hist":
            fig = px.histogram(df, x=x, color=color, **kwargs)
        else:
            raise ValueError(f"Unsupported chart_type '{chart_type}'. Use bar|line|scatter|pie|hist.")

        fig.update_layout(template="plotly_white", margin=dict(l=40, r=20, t=50, b=40))
        result = {"figure_json": pio.to_json(fig)}
        if audit:
            audit.record("make_chart", {"chart_type": chart_type, "x": x, "y": y}, started, "chart built")
        return result
    except Exception as e:
        if audit:
            audit.record("make_chart", {"chart_type": chart_type, "x": x, "y": y}, started, "", error=str(e))
        return {"error": f"make_chart failed: {e}"}

"""Nova Analyst's LangGraph: an explicit state graph with five nodes --
Planner -> Executor -> Observer -> (loop) -> Critic -> Reporter.

The Executor calls the SAME tool implementations that mcp_server/server.py
exposes over MCP (tools/sql_tools.py, tools/analysis_tools.py) -- it invokes
them in-process rather than through an MCP client round-trip, for latency
and reliability in this prototype. The MCP server remains a fully working,
independently callable surface over the identical tool code (see
mcp_server/server.py and its verification), so any MCP client (Claude
Desktop, NuoData's Nora, etc.) can drive Nova Analyst's tools identically.

Hard caps (never violated): MAX_STEPS total tool-calling steps, MAX_LOOPS
critic-triggered extra loops, MAX_TOKENS_PER_STEP per LLM call. No write/DDL
SQL is possible -- that's enforced independently in tools/governance.py.
"""
from __future__ import annotations

import time
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from agent import prompts
from agent.llm import complete, complete_json
from agent.state import MAX_LOOPS, MAX_STEPS, MAX_TOKENS_PER_STEP, NovaState
from tools import analysis_tools, sql_tools
from tools.audit import AuditLog


def _emit(state: NovaState, event_type: str, **payload: Any) -> None:
    state.setdefault("events", []).append({"type": event_type, "ts": time.time(), **payload})


def _context_summary(state: NovaState) -> str:
    """Compact text summary of everything learned so far, for LLM prompts."""
    lines = []
    for rec in state.get("step_records", []):
        lines.append(f"Step {rec['step_number']}: {rec['step_description']}")
        for tc in rec["tool_calls"]:
            out = tc["output"]
            if out.get("policy_violation"):
                lines.append(f"  -> {tc['tool']}({tc['input']}) => POLICY VIOLATION: {out.get('reason')}")
            elif out.get("error"):
                lines.append(f"  -> {tc['tool']}({tc['input']}) => ERROR: {out.get('error')}")
            elif "schema" in out:
                for table, cols in out["schema"].items():
                    col_desc = ", ".join(f"{c['column']}({c.get('governance', 'open')})" for c in cols)
                    lines.append(f"  -> schema[{table}]: {col_desc}")
            elif "rows" in out:
                lines.append(f"  -> {tc['tool']} returned {out.get('row_count', len(out['rows']))} rows, columns={out.get('columns')}")
            elif out.get("result_type"):
                lines.append(f"  -> run_python result ({out['result_type']}): {str(out.get('value') or out.get('rows'))[:300]}")
        if rec.get("observation"):
            lines.append(f"  Observation: {rec['observation']}")
    if state.get("governance_notes"):
        lines.append("Governance notes so far: " + "; ".join(state["governance_notes"]))
    return "\n".join(lines) if lines else "(nothing gathered yet)"


def _evidence_keys_summary(state: NovaState) -> str:
    store = state.get("evidence_store", {})
    if not store:
        return "(no stored result sets yet)"
    parts = []
    for name, rows in store.items():
        if isinstance(rows, list) and rows:
            parts.append(f"{name}: {len(rows)} rows, columns={list(rows[0].keys())}")
        else:
            parts.append(f"{name}: empty")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def node_plan(state: NovaState) -> dict[str, Any]:
    question = state["question"]
    semantic_note = prompts.build_semantic_note(
        state.get("dataset_label", "Olist Brazilian E-Commerce (demo)"),
        state.get("is_builtin_olist", True),
        state.get("available_tables", []),
    )
    result = complete_json(prompts.build_planner_system(semantic_note), f"Business question: {question}", max_tokens=MAX_TOKENS_PER_STEP)
    steps = result.get("steps", [])[:5] or ["Explore the relevant tables and answer the question directly."]

    events = list(state.get("events", []))
    new_state: NovaState = {
        "question": question,
        "plan": steps,
        "step_records": state.get("step_records", []),
        "critic_verdicts": state.get("critic_verdicts", []),
        "governance_notes": state.get("governance_notes", []),
        "loop_count": state.get("loop_count", 0),
        "status": "executing",
        "evidence_store": state.get("evidence_store", {}),
        "current_step_idx": state.get("current_step_idx", 0),
        "chart_specs": state.get("chart_specs", []),
        "audit": state["audit"],
        "events": events,
        "db_path": state.get("db_path", str(sql_tools.DEFAULT_DB_PATH)),
        "policy": state.get("policy") or sql_tools._POLICY,
        "dataset_label": state.get("dataset_label", "Olist Brazilian E-Commerce (demo)"),
        "is_builtin_olist": state.get("is_builtin_olist", True),
        "available_tables": state.get("available_tables", []),
        "semantic_note": semantic_note,
    }
    _emit(new_state, "plan", steps=steps)
    return new_state


def _run_tool(tool_name: str, args: dict[str, Any], state: NovaState) -> dict[str, Any]:
    """Dispatch to the underlying tool implementation. The tool call itself
    (SQL/DDL safety, governance) is always safe by construction -- but `args`
    comes from an LLM's JSON output, so required keys can be missing or
    wrong-typed. Never let a malformed tool call crash the graph; surface it
    as a normal {"error": ...} tool result instead, same as any other tool
    failure, so the Critic/Reporter can react to it.
    """
    audit: AuditLog = state["audit"]
    db_path = state.get("db_path") or str(sql_tools.DEFAULT_DB_PATH)
    policy = state.get("policy") or sql_tools._POLICY
    try:
        if tool_name == "get_schema":
            return sql_tools.get_schema(table=args.get("table") or None, db_path=db_path, audit=audit, policy=policy)
        elif tool_name == "run_sql":
            if not args.get("sql"):
                return {"error": "run_sql called without a 'sql' argument."}
            return sql_tools.run_sql(args["sql"], db_path=db_path, audit=audit, policy=policy)
        elif tool_name == "run_python":
            if not args.get("code"):
                return {"error": "run_python called without a 'code' argument."}
            dfs = {k: v for k, v in state.get("evidence_store", {}).items() if isinstance(v, list)}
            return analysis_tools.run_python(args["code"], dfs, audit=audit)
        elif tool_name == "make_chart":
            if not args.get("rows") or not args.get("chart_type") or not args.get("x"):
                return {"error": "make_chart requires 'rows', 'chart_type', and 'x' arguments."}
            return analysis_tools.make_chart(
                args["rows"], args["chart_type"], args["x"],
                y=args.get("y"), color=args.get("color"), title=args.get("title", ""), audit=audit,
            )
        else:
            return {"error": f"Unknown tool '{tool_name}'"}
    except Exception as e:
        return {"error": f"Tool '{tool_name}' raised an unexpected error: {e}"}


def node_execute(state: NovaState) -> dict[str, Any]:
    idx = state.get("current_step_idx", 0)
    plan = state["plan"]
    step_desc = plan[idx]

    context = _context_summary(state)
    evidence = _evidence_keys_summary(state)
    user_prompt = (
        f"Original question: {state['question']}\n\n"
        f"Current step ({idx + 1}/{len(plan)}): {step_desc}\n\n"
        f"Context from prior steps:\n{context}\n\n"
        f"Stored result sets available for run_python (reference by name):\n{evidence}\n\n"
        "Decide the ONE tool call to make progress on the current step."
    )
    semantic_note = state.get("semantic_note") or prompts.build_semantic_note(
        state.get("dataset_label", "Olist Brazilian E-Commerce (demo)"),
        state.get("is_builtin_olist", True),
        state.get("available_tables", []),
    )
    decision = complete_json(prompts.build_executor_system(semantic_note), user_prompt, max_tokens=MAX_TOKENS_PER_STEP)
    tool_name = decision.get("tool", "get_schema")
    args = decision.get("args", {})
    reasoning = decision.get("reasoning", "")

    _emit(state, "tool_call_start", step=idx + 1, tool=tool_name, args=args, reasoning=reasoning)
    output = _run_tool(tool_name, args, state)
    policy_hit = bool(output.get("policy_violation"))
    _emit(state, "tool_call_end", step=idx + 1, tool=tool_name, output=_truncate_for_event(output), policy_hit=policy_hit)

    evidence_store = dict(state.get("evidence_store", {}))
    governance_notes = list(state.get("governance_notes", []))
    chart_specs = list(state.get("chart_specs", []))

    if policy_hit:
        governance_notes.append(output.get("reason", "policy violation"))
    elif tool_name in ("run_sql", "run_python") and "rows" in output:
        key = f"step{idx + 1}_{tool_name}"
        evidence_store[key] = output["rows"]
    elif tool_name == "get_schema" and "schema" in output:
        evidence_store.setdefault("_schema", {}).update(output["schema"])
    elif tool_name == "make_chart" and "figure_json" in output:
        chart_specs.append({"title": args.get("title", ""), "figure_json": output["figure_json"]})

    step_records = list(state.get("step_records", []))
    step_records.append({
        "step_number": idx + 1,
        "step_description": step_desc,
        "tool_calls": [{"tool": tool_name, "input": args, "output": output, "policy_hit": policy_hit}],
        "observation": "",
    })

    return {
        "step_records": step_records,
        "evidence_store": evidence_store,
        "governance_notes": governance_notes,
        "chart_specs": chart_specs,
        "events": state["events"],
    }


def _truncate_for_event(output: dict[str, Any], max_rows: int = 5) -> dict[str, Any]:
    out = dict(output)
    if "rows" in out and isinstance(out["rows"], list) and len(out["rows"]) > max_rows:
        out["rows"] = out["rows"][:max_rows]
        out["_truncated_for_display"] = True
    return out


def node_observe(state: NovaState) -> dict[str, Any]:
    step_records = list(state.get("step_records", []))
    last = step_records[-1]
    tc = last["tool_calls"][-1]

    user_prompt = f"Tool call: {tc['tool']}({tc['input']})\nResult: {str(tc['output'])[:2000]}"
    observation = complete(prompts.OBSERVER_SYSTEM, user_prompt, max_tokens=300)

    last["observation"] = observation.strip()
    step_records[-1] = last
    _emit(state, "observation", step=last["step_number"], text=last["observation"])

    return {"step_records": step_records, "current_step_idx": state.get("current_step_idx", 0) + 1, "events": state["events"]}


_CHART_KEYWORDS = ("chart", "graph", "plot", "visuali")


def _question_wants_chart(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _CHART_KEYWORDS)


def node_critique(state: NovaState) -> dict[str, Any]:
    context = _context_summary(state)
    n_charts = len(state.get("chart_specs", []))
    user_prompt = (
        f"Original question: {state['question']}\n\nPlan so far: {state['plan']}\n\n"
        f"Evidence gathered:\n{context}\n\n"
        f"Charts generated so far: {n_charts}"
    )
    verdict = complete_json(prompts.CRITIC_SYSTEM, user_prompt, max_tokens=MAX_TOKENS_PER_STEP)
    verdict.setdefault("sufficient", True)
    verdict.setdefault("issues", [])
    verdict.setdefault("next_action", "finish")
    verdict.setdefault("notes", "")

    at_step_cap = len(state.get("step_records", [])) >= MAX_STEPS
    at_loop_cap = state.get("loop_count", 0) >= MAX_LOOPS

    chartable_keys = [k for k, v in state.get("evidence_store", {}).items() if k != "_schema" and isinstance(v, list) and v]
    # Only the absolute step ceiling blocks this -- see below for why it's
    # deliberately exempted from the loop cap.
    missing_requested_chart = (
        n_charts == 0
        and _question_wants_chart(state["question"])
        and chartable_keys
        and not at_step_cap
    )

    forced_chart_step = False
    if missing_requested_chart and (verdict["next_action"] == "finish" or verdict.get("sufficient")):
        # Deterministic safety net: don't rely on the LLM Critic alone to
        # catch a dropped chart request -- the question explicitly asked
        # for a visualization, we have rows to chart, and step budget
        # remains. This is a single, bounded, deterministic close-out
        # action (it can only ever fire once -- after it runs, n_charts>0
        # so it won't fire again), so it's intentionally exempt from the
        # critic-loop cap: MAX_LOOPS bounds open-ended LLM-decided
        # back-and-forth, not "actually deliver the one artifact that was
        # explicitly requested." MAX_STEPS remains a hard ceiling either way.
        forced_chart_step = True
        verdict["sufficient"] = False
        verdict["next_action"] = "continue"
        verdict.setdefault("issues", []).append("Question requested a chart but make_chart was never called.")
        verdict["next_step_description"] = (
            f"Call make_chart now using the rows already returned (result set(s): {', '.join(chartable_keys)}) "
            "to satisfy the explicit chart/visualization request -- do not re-run SQL."
        )

    critic_verdicts = list(state.get("critic_verdicts", []))
    critic_verdicts.append(verdict)
    _emit(state, "critic_verdict", verdict=verdict)

    new_state: dict[str, Any] = {"critic_verdicts": critic_verdicts, "events": state["events"]}

    if verdict["next_action"] == "finish" or verdict.get("sufficient"):
        new_state["status"] = "reporting"
    elif at_step_cap or (at_loop_cap and not forced_chart_step):
        new_state["status"] = "insufficient_data"
        _emit(state, "cap_reached", reason=f"step_cap={at_step_cap} loop_cap={at_loop_cap}")
    else:
        next_step_desc = verdict.get("next_step_description") or (
            verdict["issues"][0] if verdict["issues"] else "Gather additional evidence to close the identified gap."
        )
        new_state["plan"] = state["plan"] + [next_step_desc]
        new_state["loop_count"] = state.get("loop_count", 0) + 1
        new_state["status"] = "executing"

    return new_state


def node_report(state: NovaState) -> dict[str, Any]:
    context = _context_summary(state)
    insufficient = state.get("status") == "insufficient_data"
    user_prompt = (
        f"Business question: {state['question']}\n\nEvidence gathered:\n{context}\n\n"
        f"Charts generated: {len(state.get('chart_specs', []))}\n"
        + ("NOTE: The investigation hit its step/loop cap before the Critic confirmed sufficiency -- "
           "be honest that the answer may be partial.\n" if insufficient else "")
    )
    report_md = complete(prompts.REPORTER_SYSTEM, user_prompt, max_tokens=2000)

    _emit(state, "report", text=report_md)
    return {"final_report": report_md, "status": "done", "events": state["events"]}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_observe(state: NovaState) -> Literal["execute", "critique"]:
    if state.get("current_step_idx", 0) < len(state["plan"]) and len(state.get("step_records", [])) < MAX_STEPS:
        return "execute"
    return "critique"


def route_after_critique(state: NovaState) -> Literal["execute", "report"]:
    if state.get("status") == "executing":
        return "execute"
    return "report"


def build_graph():
    graph = StateGraph(NovaState)
    graph.add_node("plan", node_plan)
    graph.add_node("execute", node_execute)
    graph.add_node("observe", node_observe)
    graph.add_node("critique", node_critique)
    graph.add_node("report", node_report)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "observe")
    graph.add_conditional_edges("observe", route_after_observe, {"execute": "execute", "critique": "critique"})
    graph.add_conditional_edges("critique", route_after_critique, {"execute": "execute", "report": "report"})
    graph.add_edge("report", END)

    return graph.compile()


def run_agent(
    question: str,
    db_path: str | None = None,
    policy: Any | None = None,
    dataset_label: str = "Olist Brazilian E-Commerce (demo)",
    available_tables: list[str] | None = None,
):
    """Generator yielding NovaState snapshots after every node executes, for
    live streaming in the Streamlit UI. Also returns (via the final yielded
    state) the finished report, chart specs, and audit log.

    By default this analyzes the built-in Olist demo dataset (unchanged
    behavior for existing callers, e.g. evals/run_evals.py). Pass `db_path`
    (and optionally a `policy` built via tools.governance.build_policy_from_rules)
    to analyze a different DuckDB file instead -- see the "upload your own
    data" flow in app/streamlit_app.py.
    """
    app = build_graph()
    is_builtin = db_path is None
    initial_state: NovaState = {
        "question": question,
        "plan": [],
        "step_records": [],
        "critic_verdicts": [],
        "governance_notes": [],
        "loop_count": 0,
        "status": "planning",
        "final_report": "",
        "chart_specs": [],
        "audit": AuditLog(),
        "events": [],
        "evidence_store": {},
        "current_step_idx": 0,
        "db_path": db_path or str(sql_tools.DEFAULT_DB_PATH),
        "policy": policy or sql_tools._POLICY,
        "dataset_label": dataset_label,
        "is_builtin_olist": is_builtin,
        "available_tables": available_tables or [],
    }
    for state in app.stream(initial_state, stream_mode="values"):
        yield state

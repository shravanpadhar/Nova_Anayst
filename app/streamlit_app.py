"""Nova Analyst -- Streamlit UI. Plain-English question in, live streaming
agent trace, narrated report with charts, downloadable audit log out.

Visual design follows a Datalab-style system: Geist/Manrope typography, a
deep-blue/near-black palette, sharp minimal corners, no decorative emoji --
status is communicated with small typographic badges instead.
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.graph import run_agent  # noqa: E402
from agent.state import MAX_LOOPS, MAX_STEPS  # noqa: E402
from tools import dataset_loader, governance  # noqa: E402

st.set_page_config(page_title="Nova Analyst", layout="wide")

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

COLORS = {
    "primary": "#001DBE",
    "accent": "#2E6DFA",
    "secondary": "#E5E5F9",
    "background": "#F9FAFD",
    "surface": "#FFFFFF",
    "text": "#0F1217",
    "muted": "#5B6270",
    "border": "#E2E4EC",
    "danger_bg": "#FBEAEA",
    "danger_fg": "#B42318",
    "warning_bg": "#FFF4E5",
    "warning_fg": "#92400E",
}

PLOTLY_TEMPLATE = {
    "layout": {
        "colorway": [COLORS["primary"], COLORS["accent"], "#8892B0", "#B42318", "#92400E"],
        "font": {"family": "Manrope, sans-serif", "color": COLORS["text"]},
        "plot_bgcolor": COLORS["surface"],
        "paper_bgcolor": COLORS["surface"],
    }
}

OLIST_EXAMPLE_QUESTIONS = [
    ("What was our GMV trend over the last several months, and which product category drives the most revenue?", None),
    ("How does on-time delivery rate vary by state, and is there a relationship with review scores?", None),
    ("Which payment type do customers use most, and does installment count affect order value?", None),
    (
        "Which individual customers, identified by their unique person-level ID, have placed the most orders across their lifetime?",
        "Governance demo — this question requires a blocked column.",
    ),
]

CUSTOM_EXAMPLE_QUESTIONS = [
    ("Give me a high-level summary of this dataset -- what's in it, and what stands out?", None),
    ("Are there any obvious data quality issues (nulls, duplicates, outliers) I should know about?", None),
    ("Show me a chart of the most important trend or breakdown in this data.", None),
]


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&family=Manrope:wght@400;500;600;700&display=swap');

        :root {{
            --nova-primary: {COLORS["primary"]};
            --nova-accent: {COLORS["accent"]};
            --nova-secondary: {COLORS["secondary"]};
            --nova-bg: {COLORS["background"]};
            --nova-surface: {COLORS["surface"]};
            --nova-text: {COLORS["text"]};
            --nova-muted: {COLORS["muted"]};
            --nova-border: {COLORS["border"]};
        }}

        html, body, [class*="css"] {{
            font-family: 'Manrope', -apple-system, sans-serif;
        }}

        .stApp {{
            background: var(--nova-bg);
        }}

        h1, h2, h3, h4, h5,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 {{
            font-family: 'Geist', 'Manrope', sans-serif !important;
            color: var(--nova-text) !important;
            letter-spacing: -0.01em;
            font-weight: 700 !important;
        }}

        [data-testid="stMarkdownContainer"] table {{
            border-collapse: collapse;
            font-size: 14px;
        }}
        [data-testid="stMarkdownContainer"] table th {{
            background: var(--nova-secondary);
            color: var(--nova-primary);
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.04em;
            font-weight: 700;
            padding: 8px 12px;
            border: 1px solid var(--nova-border);
        }}
        [data-testid="stMarkdownContainer"] table td {{
            padding: 7px 12px;
            border: 1px solid var(--nova-border);
        }}
        [data-testid="stMarkdownContainer"] code {{
            font-family: 'Geist Mono', monospace;
            background: var(--nova-secondary);
            color: var(--nova-primary);
            padding: 1px 5px;
            border-radius: 2px;
            font-size: 0.85em;
        }}

        /* Buttons: secondary (near-black) by default, primary (brand blue) for kind=primary */
        .stButton > button, [data-testid="stDownloadButton"] button {{
            background: var(--nova-text);
            color: var(--nova-bg);
            border: 1px solid var(--nova-text);
            border-radius: 2px;
            font-family: 'Manrope', sans-serif;
            font-weight: 600;
            font-size: 13.5px;
            box-shadow: none;
            transition: background 0.12s ease, border-color 0.12s ease;
        }}
        .stButton > button:hover, [data-testid="stDownloadButton"] button:hover {{
            background: #262B36;
            border-color: #262B36;
            color: var(--nova-bg);
        }}
        .stButton > button[kind="primary"] {{
            background: var(--nova-primary);
            border-color: var(--nova-primary);
            color: #FCFCFC;
        }}
        .stButton > button[kind="primary"]:hover {{
            background: var(--nova-accent);
            border-color: var(--nova-accent);
        }}

        [data-testid="stTextArea"] textarea,
        [data-testid="stTextInput"] input {{
            border-radius: 2px !important;
            border-color: var(--nova-border) !important;
            font-family: 'Manrope', sans-serif;
        }}

        [data-testid="stExpander"] {{
            border: 1px solid var(--nova-border) !important;
            border-radius: 2px !important;
            background: var(--nova-surface);
        }}
        [data-testid="stExpander"] summary {{
            font-family: 'Geist Mono', monospace !important;
            font-size: 13px !important;
            font-weight: 600 !important;
            color: var(--nova-text) !important;
        }}

        [data-testid="stFileUploaderDropzone"] {{
            border-radius: 2px !important;
            border-color: var(--nova-border) !important;
            background: var(--nova-surface) !important;
        }}

        [data-testid="stDataFrame"], [data-testid="stDataEditor"] {{
            border-radius: 2px !important;
            border: 1px solid var(--nova-border) !important;
        }}

        [data-testid="stAlert"] {{
            border-radius: 2px !important;
        }}

        [data-testid="stMetric"] {{
            background: var(--nova-surface);
            border: 1px solid var(--nova-border);
            border-radius: 2px;
            padding: 12px 14px 10px 14px;
        }}
        [data-testid="stMetricLabel"] {{
            font-family: 'Geist Mono', monospace !important;
            font-size: 10.5px !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--nova-muted) !important;
        }}
        [data-testid="stMetricValue"] {{
            font-family: 'Geist', sans-serif !important;
            color: var(--nova-primary) !important;
        }}

        [data-testid="stSidebar"] {{
            background: var(--nova-surface);
            border-right: 1px solid var(--nova-border);
        }}

        /* --- Nova custom components --- */
        .nova-header {{
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 2px;
        }}
        .nova-mark {{
            width: 42px;
            height: 42px;
            min-width: 42px;
            background: var(--nova-primary);
            color: #FCFCFC;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Geist', sans-serif;
            font-weight: 700;
            font-size: 15px;
            letter-spacing: 0.01em;
            border-radius: 2px;
        }}
        .nova-title {{
            font-family: 'Geist', sans-serif;
            font-size: 30px;
            font-weight: 700;
            color: var(--nova-text);
            line-height: 1.15;
            letter-spacing: -0.01em;
        }}
        .nova-eyebrow {{
            font-family: 'Geist Mono', monospace;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: var(--nova-accent);
            font-weight: 600;
            margin: 0 0 6px 0;
        }}
        .nova-tagline {{
            font-family: 'Manrope', sans-serif;
            font-size: 15px;
            color: var(--nova-muted);
            max-width: 780px;
            margin: 12px 0 0 0;
            line-height: 1.55;
        }}
        .nova-section-title {{
            font-family: 'Geist', sans-serif;
            font-size: 19px;
            font-weight: 700;
            color: var(--nova-text);
            margin: 0 0 10px 0;
        }}
        .nova-status-card {{
            background: var(--nova-secondary);
            border-radius: 2px;
            padding: 9px 12px;
            font-size: 13px;
            color: var(--nova-text);
            margin-bottom: 10px;
            line-height: 1.5;
        }}
        .nova-status-card small {{
            color: var(--nova-muted);
            font-size: 12px;
        }}
        .nova-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 2px;
            font-family: 'Geist Mono', monospace;
            font-size: 10.5px;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            white-space: nowrap;
        }}
        .nova-step {{
            display: flex;
            gap: 14px;
            padding: 9px 0;
            border-bottom: 1px solid var(--nova-border);
            align-items: baseline;
        }}
        .nova-step:last-child {{ border-bottom: none; }}
        .nova-step-num {{
            font-family: 'Geist Mono', monospace;
            color: var(--nova-accent);
            font-weight: 600;
            font-size: 12.5px;
            min-width: 22px;
        }}
        .nova-step-text {{
            font-size: 14px;
            color: var(--nova-text);
            line-height: 1.5;
        }}
        .nova-callout {{
            background: var(--nova-surface);
            border: 1px solid var(--nova-border);
            border-left: 3px solid var(--nova-accent);
            padding: 10px 14px;
            margin: 6px 0 10px 0;
            border-radius: 2px;
        }}
        .nova-callout-label {{
            font-family: 'Geist Mono', monospace;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--nova-accent);
            font-weight: 700;
        }}
        .nova-callout p {{
            margin: 4px 0 0 0;
            font-size: 13.5px;
            color: var(--nova-text);
            line-height: 1.5;
        }}
        .nova-schema-row {{
            font-size: 13.5px;
            margin: 3px 0;
            line-height: 1.6;
        }}
        .nova-caution-note {{
            font-family: 'Geist Mono', monospace;
            font-size: 10.5px;
            color: {COLORS["danger_fg"]};
            text-transform: uppercase;
            letter-spacing: 0.02em;
            margin-top: 4px;
            display: block;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _badge(text: str, kind: str = "neutral") -> str:
    palette = {
        "neutral": (COLORS["secondary"], COLORS["primary"]),
        "accent": ("#E9EFFE", COLORS["accent"]),
        "danger": (COLORS["danger_bg"], COLORS["danger_fg"]),
        "warning": (COLORS["warning_bg"], COLORS["warning_fg"]),
        "dark": (COLORS["text"], COLORS["background"]),
    }
    bg, fg = palette.get(kind, palette["neutral"])
    return f'<span class="nova-badge" style="background:{bg};color:{fg};">{text}</span>'


def _init_state():
    st.session_state.setdefault("question_input", "")
    st.session_state.setdefault("final_state", None)
    st.session_state.setdefault("running", False)
    st.session_state.setdefault("session_id", str(uuid.uuid4())[:8])
    # Active dataset context. None db_path means "use the built-in Olist demo".
    st.session_state.setdefault("active_db_path", None)
    st.session_state.setdefault("active_policy", None)
    st.session_state.setdefault("active_dataset_label", "Olist Brazilian E-Commerce (demo)")
    st.session_state.setdefault("active_tables", [])
    # Staging area while the user reviews the auto-suggested governance policy.
    st.session_state.setdefault("pending_schema", None)
    st.session_state.setdefault("pending_db_path", None)
    st.session_state.setdefault("pending_label", None)


def _is_custom_dataset() -> bool:
    return st.session_state.get("active_db_path") is not None


def _render_tool_output(output: dict) -> None:
    if output.get("policy_violation"):
        st.markdown(_badge("Policy violation", "danger"), unsafe_allow_html=True)
        st.markdown(f"<div class='nova-schema-row'><strong>{output.get('kind')}</strong> — {output.get('reason')}</div>", unsafe_allow_html=True)
    elif output.get("error"):
        st.markdown(_badge("Tool error", "warning"), unsafe_allow_html=True)
        st.markdown(f"<div class='nova-schema-row'>{output['error']}</div>", unsafe_allow_html=True)
    elif "schema" in output:
        for table, cols in output["schema"].items():
            parts = []
            for c in cols:
                gov = c.get("governance")
                entry = f"<code>{c['column']}</code>"
                if gov == "blocked":
                    entry += " " + _badge("blocked", "danger")
                elif gov == "masked":
                    entry += " " + _badge("masked", "warning")
                parts.append(entry)
            st.markdown(f"<div class='nova-schema-row'><strong>{table}</strong> &nbsp; " + " &nbsp; ".join(parts) + "</div>", unsafe_allow_html=True)
    elif "rows" in output:
        rows = output["rows"]
        st.caption(f"{output.get('row_count', len(rows))} row(s) returned" + (" (truncated for display)" if output.get("_truncated_for_display") else ""))
        if rows:
            st.dataframe(rows, use_container_width=True, height=min(300, 40 + 35 * len(rows)))
    elif output.get("result_type"):
        st.json(output)


def _render_events(events: list[dict], seen: int) -> int:
    """Render events[seen:] into the live trace area; return new seen count."""
    for ev in events[seen:]:
        etype = ev["type"]
        if etype == "plan":
            with st.expander(f"Plan — {len(ev['steps'])} steps", expanded=True):
                for i, step in enumerate(ev["steps"], 1):
                    st.markdown(
                        f'<div class="nova-step"><span class="nova-step-num">{i:02d}</span>'
                        f'<span class="nova-step-text">{step}</span></div>',
                        unsafe_allow_html=True,
                    )
        elif etype == "tool_call_start":
            pass  # merged into tool_call_end for a cleaner trace
        elif etype == "tool_call_end":
            with st.expander(f"Step {ev['step']} · {ev['tool']}", expanded=ev.get("policy_hit", False)):
                _render_tool_output(ev["output"])
        elif etype == "observation":
            st.markdown(
                f'<div class="nova-callout"><span class="nova-callout-label">Observation</span>'
                f'<p>{ev["text"]}</p></div>',
                unsafe_allow_html=True,
            )
        elif etype == "critic_verdict":
            v = ev["verdict"]
            ok = v.get("sufficient")
            action = v.get("next_action", "?").upper()
            with st.expander(f"Critic verdict · {action}", expanded=not ok):
                st.markdown(_badge("Sufficient" if ok else "Needs another pass", "accent" if ok else "warning"), unsafe_allow_html=True)
                st.markdown(f"<div class='nova-schema-row'>{v.get('notes', '')}</div>", unsafe_allow_html=True)
                if v.get("issues"):
                    st.markdown("**Issues raised**")
                    for issue in v["issues"]:
                        st.markdown(f"- {issue}")
        elif etype == "cap_reached":
            st.info(f"Hard cap reached ({ev['reason']}) — proceeding to report with the evidence gathered so far.")
        elif etype == "report":
            pass  # rendered separately as the final report section
    return len(events)


def main():
    _init_state()
    _inject_css()
    pio.templates["nova"] = PLOTLY_TEMPLATE
    pio.templates.default = "plotly_white+nova"

    st.markdown(
        """
        <div class="nova-header">
          <div class="nova-mark">NA</div>
          <div>
            <div class="nova-eyebrow">Governed Autonomous Analytics</div>
            <div class="nova-title">Nova Analyst</div>
          </div>
        </div>
        <p class="nova-tagline">Ask a plain-English business question. Nova Analyst plans a
        multi-step investigation, queries governed tools, self-critiques its own findings, and
        returns a narrated report — with every policy decision enforced in code, not prompted.</p>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown('<div class="nova-eyebrow">Platform</div>', unsafe_allow_html=True)
        st.markdown(
            "**Nova Analyst** demonstrates:\n"
            "- Explicit LangGraph state machine (Plan → Act → Observe → Critique → Report)\n"
            "- MCP-native tools (see `mcp_server/server.py`)\n"
            "- Governance-in-the-loop (`config/governance.yaml`) — blocked/masked columns enforced in code, not prompts\n"
            "- Any-LLM swappable client (`agent/llm.py`)\n"
        )
        st.divider()
        st.markdown(f"**Caps** — max {MAX_STEPS} steps, max {MAX_LOOPS} critic-triggered loops")
        st.divider()

        st.markdown('<div class="nova-eyebrow">Data source</div>', unsafe_allow_html=True)
        if _is_custom_dataset():
            st.markdown(
                f"<div class='nova-status-card'>Active — <strong>{st.session_state['active_dataset_label']}</strong>"
                f"<br/><small>{len(st.session_state['active_tables'])} table(s): "
                f"{', '.join(st.session_state['active_tables'])}</small></div>",
                unsafe_allow_html=True,
            )
            if st.button("Switch back to Olist demo", use_container_width=True):
                st.session_state["active_db_path"] = None
                st.session_state["active_policy"] = None
                st.session_state["active_dataset_label"] = "Olist Brazilian E-Commerce (demo)"
                st.session_state["active_tables"] = []
                st.session_state["final_state"] = None
                st.rerun()
        else:
            st.markdown(
                "<div class='nova-status-card'>Active — <strong>Olist Brazilian E-Commerce (demo)</strong></div>",
                unsafe_allow_html=True,
            )

        with st.expander("Upload your own data", expanded=not _is_custom_dataset() and st.session_state.get("pending_schema") is not None):
            uploaded_files = st.file_uploader("CSV files (one table per file)", type=["csv"], accept_multiple_files=True, key="uploader")
            if uploaded_files and st.button("Analyze schema & suggest governance", use_container_width=True):
                with st.spinner("Loading files into DuckDB and scanning columns..."):
                    db_path, schema = dataset_loader.load_csvs_to_duckdb(uploaded_files, st.session_state["session_id"])
                st.session_state["pending_db_path"] = str(db_path)
                st.session_state["pending_schema"] = schema
                st.session_state["pending_label"] = ", ".join(f.name for f in uploaded_files)
                st.rerun()

    # Governance review step for an uploaded dataset that hasn't been confirmed yet.
    if st.session_state.get("pending_schema"):
        st.markdown('<div class="nova-eyebrow">Setup</div>', unsafe_allow_html=True)
        st.markdown('<div class="nova-section-title">Review governance policy for your data</div>', unsafe_allow_html=True)
        st.caption(
            "Nova Analyst auto-suggested a policy from column names (heuristic, not a real data "
            "classification). Review and adjust before running the agent -- this is the same "
            "block/mask/open mechanism used for the built-in demo's governance.yaml."
        )
        suggestions = governance.suggest_policy_for_schema(st.session_state["pending_schema"])
        df = pd.DataFrame(suggestions)
        edited = st.data_editor(
            df,
            column_config={
                "table": st.column_config.TextColumn(disabled=True),
                "column": st.column_config.TextColumn(disabled=True),
                "policy": st.column_config.SelectboxColumn(options=["open", "masked", "blocked"], required=True),
                "reason": st.column_config.TextColumn(),
            },
            hide_index=True,
            use_container_width=True,
            key="governance_editor",
        )
        c1, c2 = st.columns([1, 4])
        if c1.button("Confirm & activate dataset", type="primary"):
            st.session_state["active_policy"] = governance.build_policy_from_rules(edited.to_dict(orient="records"))
            st.session_state["active_db_path"] = st.session_state["pending_db_path"]
            st.session_state["active_dataset_label"] = st.session_state["pending_label"]
            st.session_state["active_tables"] = list(st.session_state["pending_schema"].keys())
            st.session_state["pending_schema"] = None
            st.session_state["pending_db_path"] = None
            st.session_state["pending_label"] = None
            st.session_state["final_state"] = None
            st.session_state["question_input"] = ""
            st.rerun()
        if c2.button("Cancel"):
            st.session_state["pending_schema"] = None
            st.session_state["pending_db_path"] = None
            st.session_state["pending_label"] = None
            st.rerun()
        st.divider()

    is_custom = _is_custom_dataset()
    example_questions = CUSTOM_EXAMPLE_QUESTIONS if is_custom else OLIST_EXAMPLE_QUESTIONS

    st.markdown('<div class="nova-eyebrow">Investigate</div>', unsafe_allow_html=True)
    st.markdown('<div class="nova-section-title">Ask a question</div>', unsafe_allow_html=True)
    cols = st.columns(len(example_questions))
    for i, (q, note) in enumerate(example_questions):
        label = q if len(q) < 60 else q[:57] + "..."
        tooltip = q + (f"\n\n{note}" if note else "")
        if cols[i].button(label, key=f"ex_{i}", use_container_width=True, help=tooltip):
            st.session_state["question_input"] = q
        if note:
            cols[i].markdown(f"<span class='nova-caution-note'>{note}</span>", unsafe_allow_html=True)

    question = st.text_area("Business question", key="question_input", height=80, placeholder="e.g. What drove the drop in GMV last quarter?")
    run_clicked = st.button("Run Nova Analyst", type="primary", disabled=st.session_state["running"])

    trace_container = st.container()

    if run_clicked and question.strip():
        st.session_state["running"] = True
        st.session_state["final_state"] = None
        seen = 0
        start_time = time.time()

        with trace_container:
            st.markdown('<div class="nova-eyebrow">Agent trace</div>', unsafe_allow_html=True)
            st.markdown('<div class="nova-section-title">Live investigation</div>', unsafe_allow_html=True)
            trace_area = st.container()
            status_ph = st.empty()

            final_state = None
            try:
                for state in run_agent(
                    question.strip(),
                    db_path=st.session_state.get("active_db_path"),
                    policy=st.session_state.get("active_policy"),
                    dataset_label=st.session_state.get("active_dataset_label", "Olist Brazilian E-Commerce (demo)"),
                    available_tables=st.session_state.get("active_tables", []),
                ):
                    with trace_area:
                        seen = _render_events(state.get("events", []), seen)
                    status_ph.info(f"Status: `{state.get('status')}` — {len(state.get('step_records', []))} step(s) executed so far...")
                    final_state = state
            except Exception as e:
                st.error(f"Agent run failed: {e}")
                st.session_state["running"] = False
                st.stop()

            elapsed = time.time() - start_time
            status_ph.success(f"Done in {elapsed:.1f}s — status: `{final_state.get('status')}`")
            st.session_state["final_state"] = final_state
            st.session_state["elapsed"] = elapsed

        st.session_state["running"] = False

    final_state = st.session_state.get("final_state")
    if final_state:
        st.divider()
        st.markdown('<div class="nova-eyebrow">Output</div>', unsafe_allow_html=True)
        st.markdown('<div class="nova-section-title">Report</div>', unsafe_allow_html=True)
        st.markdown(final_state.get("final_report", "_No report generated._"))

        charts = final_state.get("chart_specs", [])
        if charts:
            st.markdown('<div class="nova-section-title">Charts</div>', unsafe_allow_html=True)
            for spec in charts:
                fig = go.Figure(json.loads(spec["figure_json"]))
                fig.update_layout(template="plotly_white+nova")
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        audit = final_state["audit"]
        summary = audit.summary()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Tool calls", summary["total_tool_calls"])
        m2.metric("Tool errors", summary["tool_errors"])
        m3.metric("Policy hits", summary["policy_hits"])
        m4.metric("Avg latency (ms)", summary["avg_latency_ms"])
        m5.metric("Critic loops", final_state.get("loop_count", 0))

        st.download_button(
            "Download audit log (JSON)",
            data=audit.to_json(),
            file_name=f"nova_analyst_audit_{int(time.time())}.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()

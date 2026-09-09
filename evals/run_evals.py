"""Eval harness for Nova Analyst. Runs every question in questions.yaml
through the full agent graph and writes evals/RESULTS.md with:
  - task success rate
  - avg steps per question
  - tool-error rate
  - % of runs where the Critic triggered at least one self-correction loop

Scoring method (documented honestly, not dressed up as an LLM judge):
a question is scored PASS by simple, transparent rules -- see `score_question`
below. This is a keyword/structural check against the final report, not a
black-box judge call, so every score is reproducible and auditable.

Usage:
    python evals/run_evals.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.graph import run_agent  # noqa: E402

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.yaml"
RESULTS_PATH = Path(__file__).resolve().parent / "RESULTS.md"


def score_question(q: dict, final_state: dict) -> tuple[bool, str]:
    """Transparent, rule-based scoring -- see module docstring."""
    report = (final_state.get("final_report") or "").lower()
    if not report or len(report) < 100:
        return False, "No substantive report was produced."

    if q.get("expects_governance_hit"):
        governance_fired = bool(final_state.get("governance_notes"))
        keyword_hit = any(kw in report for kw in q.get("expected_keywords", []))
        if not (governance_fired or keyword_hit):
            return False, "Question required surfacing a governance limitation, but none was found in governance_notes or the report."
    else:
        missing = [kw for kw in q.get("expected_keywords", []) if kw.lower() not in report]
        if missing:
            return False, f"Report is missing expected keyword(s): {missing}"

    if q.get("expects_chart") and not final_state.get("chart_specs"):
        return False, "Question asked for a chart but none was generated."

    if final_state.get("status") not in ("done",):
        return False, f"Run ended in status '{final_state.get('status')}' rather than 'done'."

    return True, "OK"


def run_all() -> list[dict]:
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = yaml.safe_load(f)["questions"]

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Running {q['id']} ({q['difficulty']}): {q['question'][:70]}...")
        start = time.time()
        final_state = None
        error = None
        try:
            for state in run_agent(q["question"]):
                final_state = state
        except Exception as e:
            error = str(e)
        elapsed = time.time() - start

        if error or final_state is None:
            results.append({
                "id": q["id"], "difficulty": q["difficulty"], "question": q["question"],
                "success": False, "notes": f"Run raised an exception: {error}",
                "steps": 0, "loops": 0, "tool_calls": 0, "tool_errors": 0,
                "policy_hits": 0, "elapsed_s": round(elapsed, 1), "self_corrected": False,
            })
            print(f"    -> FAILED (exception): {error}")
            continue

        success, notes = score_question(q, final_state)
        audit_summary = final_state["audit"].summary()
        loop_count = final_state.get("loop_count", 0)

        results.append({
            "id": q["id"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "success": success,
            "notes": notes,
            "steps": len(final_state.get("step_records", [])),
            "loops": loop_count,
            "tool_calls": audit_summary["total_tool_calls"],
            "tool_errors": audit_summary["tool_errors"],
            "policy_hits": audit_summary["policy_hits"],
            "elapsed_s": round(elapsed, 1),
            "self_corrected": loop_count > 0,
        })
        print(f"    -> {'PASS' if success else 'FAIL'} ({elapsed:.1f}s, {len(final_state.get('step_records', []))} steps, {loop_count} loop(s))")

    return results


def write_results_md(results: list[dict]) -> None:
    n = len(results)
    n_success = sum(1 for r in results if r["success"])
    avg_steps = sum(r["steps"] for r in results) / n if n else 0
    total_tool_calls = sum(r["tool_calls"] for r in results)
    total_tool_errors = sum(r["tool_errors"] for r in results)
    tool_error_rate = (total_tool_errors / total_tool_calls) if total_tool_calls else 0
    pct_self_corrected = sum(1 for r in results if r["self_corrected"]) / n if n else 0
    total_policy_hits = sum(r["policy_hits"] for r in results)

    lines = []
    lines.append("# Nova Analyst -- Eval Results\n")
    lines.append(f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    success_pct = (n_success / n * 100) if n else 0
    lines.append(f"| Task success rate | **{n_success}/{n} ({success_pct:.0f}%)** |")
    lines.append(f"| Avg steps per question | {avg_steps:.1f} |")
    lines.append(f"| Tool-error rate | {tool_error_rate*100:.1f}% ({total_tool_errors}/{total_tool_calls} calls) |")
    lines.append(f"| Runs with a Critic-triggered self-correction | {pct_self_corrected*100:.0f}% |")
    lines.append(f"| Total governance policy hits across all runs | {total_policy_hits} |")
    lines.append("")
    lines.append("## Per-question results\n")
    lines.append("| ID | Difficulty | Question | Result | Steps | Loops | Tool errors | Policy hits | Time (s) | Notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        icon = "✅" if r["success"] else "❌"
        q_short = r["question"] if len(r["question"]) < 70 else r["question"][:67] + "..."
        lines.append(
            f"| {r['id']} | {r['difficulty']} | {q_short} | {icon} | {r['steps']} | {r['loops']} | "
            f"{r['tool_errors']} | {r['policy_hits']} | {r['elapsed_s']} | {r['notes']} |"
        )
    lines.append("")
    lines.append("## Scoring methodology (honest disclosure)\n")
    lines.append(
        "Scoring is rule-based, not an LLM judge: each question passes if (1) a "
        "substantive final report was produced, (2) for questions expecting a "
        "governance limitation, the run actually recorded a policy hit or the "
        "report names the restriction, (3) for other questions, the report "
        "contains a small set of expected keywords, (4) if a chart was "
        "requested, one was generated, and (5) the run reached status `done`. "
        "This is intentionally simple and fully reproducible -- see "
        "`evals/run_evals.py::score_question` for the exact logic. It checks "
        "for topical coverage and honest governance behavior, not numeric "
        "correctness of every figure in the report."
    )
    lines.append("")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")


def main():
    results = run_all()
    write_results_md(results)


if __name__ == "__main__":
    main()

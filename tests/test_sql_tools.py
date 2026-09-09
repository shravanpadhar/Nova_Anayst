"""Unit tests for tools/sql_tools.py against the loaded DuckDB warehouse.
Requires data/olist.duckdb to exist (run `python scripts/load_data.py` first).
"""
import pytest

from tools import sql_tools
from tools.audit import AuditLog


@pytest.fixture
def audit():
    return AuditLog()


def test_list_tables_returns_expected_tables(audit):
    result = sql_tools.list_tables(audit=audit)
    names = {t["table"] for t in result["tables"]}
    assert {"orders", "customers", "order_items", "reviews"}.issubset(names)
    assert all(t["row_count"] >= 0 for t in result["tables"])


def test_get_schema_annotates_governed_columns(audit):
    result = sql_tools.get_schema(table="customers", audit=audit)
    cols = {c["column"]: c for c in result["schema"]["customers"]}
    assert cols["customer_unique_id"]["governance"] == "blocked"
    assert "governance" not in cols["customer_id"]


def test_run_sql_simple_select_works(audit):
    result = sql_tools.run_sql("SELECT COUNT(*) AS n FROM orders", audit=audit)
    assert "rows" in result
    assert result["rows"][0]["n"] >= 0


def test_run_sql_blocks_governed_column(audit):
    result = sql_tools.run_sql("SELECT customer_unique_id, COUNT(*) FROM customers GROUP BY 1", audit=audit)
    assert result.get("policy_violation") is True
    assert result["kind"] == "blocked_column"


def test_run_sql_blocks_select_star_on_governed_table(audit):
    result = sql_tools.run_sql("SELECT * FROM customers LIMIT 5", audit=audit)
    assert result.get("policy_violation") is True


def test_run_sql_masks_review_comments(audit):
    result = sql_tools.run_sql(
        "SELECT review_comment_message FROM reviews WHERE review_comment_message IS NOT NULL LIMIT 5",
        audit=audit,
    )
    assert "rows" in result
    for row in result["rows"]:
        assert row["review_comment_message"] == "[REDACTED]"


def test_run_sql_denies_write_statements(audit):
    result = sql_tools.run_sql("DELETE FROM orders", audit=audit)
    assert result.get("policy_violation") is True
    assert result["kind"] == "write_denied"


def test_run_sql_denies_ddl(audit):
    result = sql_tools.run_sql("DROP TABLE orders", audit=audit)
    assert result.get("policy_violation") is True


def test_run_sql_invalid_sql_returns_policy_violation_or_error(audit):
    result = sql_tools.run_sql("SELECT FROM WHERE ???", audit=audit)
    assert result.get("policy_violation") is True or "error" in result


def test_audit_log_records_calls(audit):
    sql_tools.run_sql("SELECT 1", audit=audit)
    entries = audit.entries()
    assert len(entries) == 1
    assert entries[0].tool_name == "run_sql"

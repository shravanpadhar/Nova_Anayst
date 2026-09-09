"""Unit tests for tools/governance.py -- the riskiest module, since a bug
here means governed data leaks."""
import pytest

from tools import governance


@pytest.fixture
def policy():
    return governance.load_policy()


def test_policy_loads_blocked_and_masked(policy):
    assert ("customers", "customer_unique_id") in policy.blocked_set()
    assert ("reviews", "review_comment_message") in policy.masked_set()


def test_check_statement_type_allows_select(policy):
    governance.check_statement_type("SELECT order_id FROM orders LIMIT 5")  # should not raise


def test_check_statement_type_allows_cte(policy):
    governance.check_statement_type("WITH t AS (SELECT 1 AS x) SELECT * FROM t")  # should not raise


@pytest.mark.parametrize("sql", [
    "DROP TABLE orders",
    "DELETE FROM orders WHERE order_id = 'x'",
    "UPDATE orders SET order_status = 'x'",
    "INSERT INTO orders VALUES (1)",
    "ALTER TABLE orders ADD COLUMN x INT",
    "ATTACH 'foo.db' AS foo",
])
def test_check_statement_type_denies_writes(sql):
    with pytest.raises(governance.PolicyViolation):
        governance.check_statement_type(sql)


def test_blocked_column_direct_reference(policy):
    hits = governance.find_blocked_column_refs(
        "SELECT customer_unique_id FROM customers", policy
    )
    assert ("customers", "customer_unique_id") in hits


def test_blocked_column_qualified_reference(policy):
    hits = governance.find_blocked_column_refs(
        "SELECT c.customer_unique_id FROM customers c", policy
    )
    assert ("customers", "customer_unique_id") in hits


def test_blocked_column_select_star(policy):
    hits = governance.find_blocked_column_refs(
        "SELECT * FROM customers", policy
    )
    assert ("customers", "customer_unique_id") in hits


def test_safe_query_has_no_blocked_hits(policy):
    hits = governance.find_blocked_column_refs(
        "SELECT customer_id, customer_city FROM customers", policy
    )
    assert hits == []


def test_unrelated_table_same_column_name_not_flagged(policy):
    # A different table happening to have a column with the same name as a
    # blocked one, but the blocked table isn't in the query, should not fire.
    hits = governance.find_blocked_column_refs(
        "SELECT order_id FROM orders", policy
    )
    assert hits == []


def test_annotate_schema_marks_blocked_and_masked(policy):
    schema = {
        "customers": [{"column": "customer_id", "type": "VARCHAR"}, {"column": "customer_unique_id", "type": "VARCHAR"}],
        "reviews": [{"column": "review_score", "type": "INTEGER"}, {"column": "review_comment_message", "type": "VARCHAR"}],
    }
    annotated = governance.annotate_schema(schema, policy)
    cust_cols = {c["column"]: c for c in annotated["customers"]}
    rev_cols = {c["column"]: c for c in annotated["reviews"]}

    assert cust_cols["customer_unique_id"]["governance"] == "blocked"
    assert "governance" not in cust_cols["customer_id"]
    assert rev_cols["review_comment_message"]["governance"] == "masked"
    assert "governance" not in rev_cols["review_score"]


def test_mask_dataframe_columns_redacts_values(policy):
    import pandas as pd
    df = pd.DataFrame({"review_comment_message": ["hello", "world"], "review_score": [5, 4]})
    masked = governance.mask_dataframe_columns(df, {}, policy)
    assert (masked["review_comment_message"] == "[REDACTED]").all()
    assert list(masked["review_score"]) == [5, 4]

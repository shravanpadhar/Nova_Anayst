"""Governance enforcement: loads config/governance.yaml and mechanically
enforces it against schema reads and SQL queries. This is the module that
makes "governance-in-the-loop" real rather than a prompt instruction the LLM
could ignore -- every check here runs in plain Python before/after DuckDB.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlglot
import sqlglot.expressions as exp
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "governance.yaml"


@dataclass
class ColumnRule:
    table: str
    column: str
    reason: str
    mask_strategy: str | None = None


@dataclass
class GovernancePolicy:
    version: int
    blocked: list[ColumnRule] = field(default_factory=list)
    masked: list[ColumnRule] = field(default_factory=list)
    allow_only: list[str] = field(default_factory=lambda: ["SELECT", "WITH"])
    deny_statements: list[str] = field(default_factory=list)
    max_rows_returned: int = 5000

    def blocked_set(self) -> set[tuple[str, str]]:
        return {(r.table.lower(), r.column.lower()) for r in self.blocked}

    def masked_set(self) -> set[tuple[str, str]]:
        return {(r.table.lower(), r.column.lower()) for r in self.masked}

    def blocked_reason(self, table: str, column: str) -> str | None:
        for r in self.blocked:
            if r.table.lower() == table.lower() and r.column.lower() == column.lower():
                return r.reason
        return None


def load_policy(path: Path = CONFIG_PATH) -> GovernancePolicy:
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    blocked = [ColumnRule(table=c["table"], column=c["column"], reason=c.get("reason", "")) for c in raw.get("blocked_columns", [])]
    masked = [
        ColumnRule(table=c["table"], column=c["column"], reason=c.get("reason", ""), mask_strategy=c.get("mask_strategy", "redact"))
        for c in raw.get("masked_columns", [])
    ]
    sql_policy = raw.get("sql_policy", {})
    return GovernancePolicy(
        version=raw.get("policy_version", 1),
        blocked=blocked,
        masked=masked,
        allow_only=sql_policy.get("allow_only", ["SELECT", "WITH"]),
        deny_statements=sql_policy.get("deny_statements", []),
        max_rows_returned=sql_policy.get("max_rows_returned", 5000),
    )


# Heuristic PII patterns used to auto-suggest a governance policy for a
# user-uploaded dataset that has no config/governance.yaml written for it.
# Order matters -- first match wins. This is a starting point for the user
# to review/edit in the UI, not a substitute for real data classification.
_PII_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"(ssn|social_security|national_id|tax_id)", re.I), "blocked", "Looks like a government identifier."),
    (re.compile(r"passport", re.I), "blocked", "Looks like a government identifier."),
    (re.compile(r"(credit_card|card_number|\bcvv\b|iban|account_number|bank_account|routing_number)", re.I), "blocked", "Looks like a financial account identifier."),
    (re.compile(r"(unique_id|person_id|customer_unique)", re.I), "blocked", "Looks like a stable person-level identifier that could re-identify individuals."),
    (re.compile(r"\bemail\b", re.I), "masked", "Looks like a direct contact identifier (email)."),
    (re.compile(r"(phone|mobile|\btel\b)", re.I), "masked", "Looks like a direct contact identifier (phone)."),
    (re.compile(r"(first_name|last_name|full_name|customer_name|person_name|contact_name|^name$)", re.I), "masked", "Looks like a personal name."),
    (re.compile(r"(address|street)", re.I), "masked", "Looks like a physical address."),
    (re.compile(r"(dob|date_of_birth|birth_date)", re.I), "masked", "Looks like a date of birth."),
    (re.compile(r"(ip_address|^ip$)", re.I), "masked", "Looks like a device/network identifier."),
    (re.compile(r"(comment|review_text|message|notes?$|description|feedback)", re.I), "masked", "Free-text field may contain names, contact info, or other PII."),
]


def suggest_policy_for_schema(schema: dict[str, list[dict]]) -> list[dict]:
    """Best-effort, name-pattern-only suggestion of a governance policy for
    an arbitrary uploaded dataset (no ground truth -- just column-name
    heuristics). Returns one row per column: {table, column, policy, reason}
    with policy in "open"|"masked"|"blocked", meant to be reviewed and
    edited by the user before being turned into a real GovernancePolicy via
    build_policy_from_rules().
    """
    suggestions = []
    for table, cols in schema.items():
        for c in cols:
            col = c["column"]
            level, reason = "open", ""
            for pattern, matched_level, matched_reason in _PII_PATTERNS:
                if pattern.search(col):
                    level, reason = matched_level, matched_reason
                    break
            suggestions.append({"table": table, "column": col, "policy": level, "reason": reason})
    return suggestions


def build_policy_from_rules(rules: list[dict]) -> "GovernancePolicy":
    """Build a GovernancePolicy from user-reviewed rows (as produced by
    suggest_policy_for_schema and edited in the UI): {table, column, policy, reason}.
    """
    blocked = [
        ColumnRule(table=r["table"], column=r["column"], reason=r.get("reason") or "Flagged as blocked by dataset owner.")
        for r in rules if r.get("policy") == "blocked"
    ]
    masked = [
        ColumnRule(table=r["table"], column=r["column"], reason=r.get("reason") or "Flagged as masked by dataset owner.", mask_strategy="redact")
        for r in rules if r.get("policy") == "masked"
    ]
    return GovernancePolicy(
        version=1,
        blocked=blocked,
        masked=masked,
        allow_only=["SELECT", "WITH"],
        deny_statements=["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "COPY", "EXPORT", "PRAGMA", "CALL"],
        max_rows_returned=5000,
    )


class PolicyViolation(Exception):
    """Raised internally; callers should catch and convert to a structured dict."""

    def __init__(self, reason: str, kind: str = "blocked_column"):
        self.reason = reason
        self.kind = kind
        super().__init__(reason)


def check_statement_type(sql: str) -> None:
    """Reject anything that isn't a read-only SELECT/WITH/CTE statement."""
    try:
        parsed = sqlglot.parse_one(sql, read="duckdb")
    except Exception as e:
        raise PolicyViolation(f"SQL could not be parsed, refusing to execute: {e}", kind="parse_error")

    deny_types = {
        exp.Insert: "INSERT", exp.Update: "UPDATE", exp.Delete: "DELETE",
        exp.Drop: "DROP", exp.Alter: "ALTER", exp.Create: "CREATE",
        exp.Attach: "ATTACH", exp.Copy: "COPY", exp.Pragma: "PRAGMA",
    }
    for node in parsed.walk():
        node_obj = node[0] if isinstance(node, tuple) else node
        for cls, name in deny_types.items():
            if isinstance(node_obj, cls):
                raise PolicyViolation(f"Statement type '{name}' is not permitted -- read-only SELECT/WITH only.", kind="write_denied")

    root_type = type(parsed)
    if not isinstance(parsed, (exp.Select, exp.Union, exp.With)):
        raise PolicyViolation(f"Only SELECT/WITH statements are permitted, got {root_type.__name__}.", kind="write_denied")


def find_blocked_column_refs(sql: str, policy: GovernancePolicy) -> list[tuple[str, str]]:
    """Best-effort static check: flag any column reference matching a blocked
    (table, column) pair, including `SELECT *`. This is intentionally
    conservative -- if a blocked column NAME appears anywhere in the query
    (any table alias, unqualified, or via *), we flag it rather than risk a
    false negative that leaks governed data.
    """
    try:
        parsed = sqlglot.parse_one(sql, read="duckdb")
    except Exception:
        return []

    blocked = policy.blocked_set()
    blocked_col_names = {col for (_, col) in blocked}
    hits: set[tuple[str, str]] = set()

    star_used = any(isinstance(n[0] if isinstance(n, tuple) else n, exp.Star) for n in parsed.walk())
    tables_in_query = {t.name.lower() for t in parsed.find_all(exp.Table)}

    if star_used:
        for (tbl, col) in blocked:
            if tbl.lower() in tables_in_query:
                hits.add((tbl, col))

    for col_node in parsed.find_all(exp.Column):
        col_name = col_node.name.lower() if col_node.name else ""
        if col_name in blocked_col_names:
            for (tbl, col) in blocked:
                if col.lower() == col_name and tbl.lower() in tables_in_query:
                    hits.add((tbl, col))

    return sorted(hits)


def annotate_schema(schema: dict[str, list[dict]], policy: GovernancePolicy) -> dict[str, list[dict]]:
    """Given {table: [{column, type}, ...]}, annotate/hide governed columns."""
    blocked = policy.blocked_set()
    masked = policy.masked_set()
    out: dict[str, list[dict]] = {}
    for table, cols in schema.items():
        new_cols = []
        for c in cols:
            key = (table.lower(), c["column"].lower())
            entry = dict(c)
            if key in blocked:
                entry["governance"] = "blocked"
                entry["note"] = policy.blocked_reason(table, c["column"])
            elif key in masked:
                entry["governance"] = "masked"
            new_cols.append(entry)
        out[table] = new_cols
    return out


def mask_dataframe_columns(df, table_hint_columns: dict[str, str], policy: GovernancePolicy):
    """Redact values in any DataFrame column whose name matches a masked
    column rule. `table_hint_columns` maps result-column-name -> source table
    (best-effort, from the query's FROM/JOIN) so masking matches by name.
    """
    masked_names = {col for (_, col) in policy.masked_set()}
    for col in list(df.columns):
        if col.lower() in masked_names:
            df[col] = "[REDACTED]"
    return df

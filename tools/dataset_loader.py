"""Loads user-uploaded CSV files into a fresh DuckDB file so the agent can
analyze arbitrary datasets, not just the built-in Olist demo. Used by the
"Upload your own data" flow in app/streamlit_app.py.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import duckdb

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "data" / "uploads"


def sanitize_table_name(filename: str) -> str:
    stem = Path(filename).stem
    name = re.sub(r"[^a-zA-Z0-9_]", "_", stem).strip("_").lower() or "table"
    if name[0].isdigit():
        name = f"t_{name}"
    return name


def load_csvs_to_duckdb(files: list[Any], session_id: str) -> tuple[Path, dict[str, list[dict]]]:
    """Load one or more uploaded CSV files into a new DuckDB file.

    `files`: objects with `.name` and `.getvalue()` (Streamlit's
    UploadedFile) or (filename, bytes) tuples -- either works.
    `session_id`: used to namespace the DuckDB file and raw CSV copies so
    concurrent Streamlit sessions don't collide.

    Returns (db_path, schema) where schema is {table: [{"column", "type"}]},
    the same shape tools.sql_tools.get_schema returns (pre-governance).
    """
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = UPLOAD_DIR / f"{session_id}_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    db_path = UPLOAD_DIR / f"{session_id}_{int(time.time())}.duckdb"

    con = duckdb.connect(str(db_path))
    used_names: set[str] = set()
    schema: dict[str, list[dict]] = {}

    try:
        for f in files:
            name = getattr(f, "name", None) or f[0]
            data = f.getvalue() if hasattr(f, "getvalue") else f[1]

            table = sanitize_table_name(name)
            base, i = table, 2
            while table in used_names:
                table = f"{base}_{i}"
                i += 1
            used_names.add(table)

            csv_path = raw_dir / f"{table}.csv"
            csv_path.write_bytes(data)

            con.execute(
                f"CREATE OR REPLACE TABLE {table} AS "
                "SELECT * FROM read_csv_auto(?, ignore_errors=true, sample_size=-1)",
                [str(csv_path)],
            )
            cols = con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'main' AND table_name = ? ORDER BY ordinal_position",
                [table],
            ).fetchall()
            schema[table] = [{"column": c, "type": t} for c, t in cols]
    finally:
        con.close()

    return db_path, schema

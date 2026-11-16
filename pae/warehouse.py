"""Loading an event stream into DuckDB, from synthetic or real sources.

DuckDB rather than a cloud warehouse because the whole project has to run on a laptop and in CI with
no credentials. The SQL is ordinary enough that the models move to BigQuery or Snowflake with little
more than a dialect change, and `scripts/run_dbt.sh` is where that swap would happen.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_DB = Path("data/warehouse.duckdb")
RAW_TABLES = ("users", "assignment", "events", "ground_truth")


def connect(path: str | Path = DEFAULT_DB, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def load(tables: dict[str, pd.DataFrame], path: str | Path = DEFAULT_DB,
         schema: str = "raw") -> dict[str, int]:
    """Replace the raw schema with these frames. Idempotent, so a reseed is never a half-load."""
    counts = {}
    with connect(path) as con:
        # Each statement gets its own cursor. DuckDB holds an unconsumed result open on the
        # connection, so reusing one connection for the next statement raises "connection has an
        # unfinished result" — including after DDL, which looks like it returns nothing.
        con.cursor().execute(f"create schema if not exists {schema}").fetchall()
        for name, frame in tables.items():
            cursor = con.cursor()
            cursor.register("_incoming", frame)
            cursor.execute(
                f"create or replace table {schema}.{name} as select * from _incoming").fetchall()
            cursor.unregister("_incoming")
            counts[name] = int(con.cursor().execute(
                f"select count(*) from {schema}.{name}").fetchone()[0])
    return counts


def query(sql: str, path: str | Path = DEFAULT_DB, **params) -> pd.DataFrame:
    """Run SQL and return a frame. Parameters are substituted with DuckDB's `$name` binding rather
    than f-strings, so a value can never be read as SQL."""
    with connect(path, read_only=True) as con:
        cursor = con.cursor()
        return cursor.execute(sql, params).df() if params else cursor.execute(sql).df()


def tables(path: str | Path = DEFAULT_DB) -> pd.DataFrame:
    return query("""
        select schema_name, table_name, estimated_size as approx_rows
        from duckdb_tables() order by schema_name, table_name
    """, path)


def views(path: str | Path = DEFAULT_DB) -> pd.DataFrame:
    """Views, listed separately from tables because the staging layer is materialised as views.

    Staging is a view on purpose: it is a thin typing-and-de-duplication pass with no expensive
    work, so materialising it would double the storage and add a refresh step for nothing.
    """
    return query("""
        select schema_name, view_name
        from duckdb_views() where not internal order by schema_name, view_name
    """, path)


def run_sql_file(sql_path: str | Path, path: str | Path = DEFAULT_DB) -> pd.DataFrame:
    """Execute a .sql file and return the final statement's result."""
    statements = [s.strip() for s in Path(sql_path).read_text().split(";") if s.strip()]
    with connect(path, read_only=True) as con:
        frame = pd.DataFrame()
        for statement in statements:
            frame = con.cursor().execute(statement).df()
        return frame

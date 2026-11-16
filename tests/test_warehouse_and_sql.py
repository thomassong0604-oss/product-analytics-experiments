"""The warehouse: loading, and the invariants the dbt models and standalone SQL must hold."""
import pandas as pd
import pytest

from pae.warehouse import load, query, run_sql_file, tables as list_tables, views as list_views
from tests.conftest import ROOT


def test_loading_is_idempotent(tmp_path, tables):
    db = tmp_path / "w.duckdb"
    first = load(tables, db)
    second = load(tables, db)
    assert first == second
    assert query("select count(*) as n from raw.events", db).n.iloc[0] == first["events"]


def test_query_parameters_are_bound_not_interpolated(tmp_path, tables):
    """A value must never be able to act as SQL."""
    db = tmp_path / "w.duckdb"
    load(tables, db)
    result = query("select count(*) as n from raw.events where event_name = $name",
                   db, name="purchase")
    assert result.n.iloc[0] > 0
    hostile = query("select count(*) as n from raw.events where event_name = $name",
                    db, name="purchase'; drop table raw.events; --")
    assert hostile.n.iloc[0] == 0
    assert query("select count(*) as n from raw.events", db).n.iloc[0] > 0


def test_the_models_all_materialise(warehouse):
    """Staging is materialised as views and everything downstream as tables, so both catalogs have
    to be checked. Looking only at tables silently passes while the staging layer is missing."""
    tables_present = set(list_tables(warehouse).table_name)
    views_present = set(list_views(warehouse).view_name)
    assert {"stg_events", "stg_users", "stg_assignment"} <= views_present
    assert {"int_sessions", "int_user_period", "fct_user_metrics", "mart_funnel",
            "mart_retention", "mart_experiment_daily"} <= tables_present


def test_duplicate_events_are_removed_by_staging(tmp_path_factory, tables):
    """De-duplication lives in staging so a double-counted purchase cannot reach a metric. Feeding
    it a known duplicate is the only way to know the rule is actually doing something."""
    import os
    import shutil
    import subprocess
    if shutil.which("dbt") is None:
        pytest.skip("dbt is not installed")
    db = tmp_path_factory.mktemp("dupes") / "test.duckdb"
    doubled = dict(tables)
    doubled["events"] = pd.concat([tables["events"], tables["events"].head(500)],
                                  ignore_index=True)
    load(doubled, db)
    profile = (ROOT / "dbt" / "profiles.yml").read_text().replace(
        'path: "../data/warehouse.duckdb"', f'path: "{db}"')
    (db.parent / "profiles.yml").write_text(profile)
    environment = {**os.environ, "DBT_PROFILES_DIR": str(db.parent),
                   "DBT_TARGET_PATH": str(db.parent / "t"), "DBT_LOG_PATH": str(db.parent / "l")}
    result = subprocess.run(["dbt", "build", "--no-use-colors"],
                            cwd=ROOT / "dbt", env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:]
    raw_rows = query("select count(*) as n from raw.events", db).n.iloc[0]
    staged = query("select count(*) as n from main_staging.stg_events", db).n.iloc[0]
    assert raw_rows == len(tables["events"]) + 500
    assert staged == len(tables["events"]), "staging should collapse the duplicates"


def test_the_funnel_only_narrows(warehouse):
    funnel = query("select * from main_marts.mart_funnel order by variant, step_number", warehouse)
    for _, group in funnel.groupby("variant"):
        assert group.reached.is_monotonic_decreasing


def test_week_zero_retention_is_complete(warehouse):
    """Every user is active in their own signup week by construction. A value below 1 means the
    cohort denominator is being taken from the active subset instead of the whole cohort."""
    retention = query("""
        select retention from main_marts.mart_retention where weeks_since_signup = 0
    """, warehouse)
    assert (retention.retention == 1.0).all()


def test_retention_decays(warehouse):
    retention = query("""
        select weeks_since_signup, avg(retention) as retention
        from main_marts.mart_retention group by 1 order by 1
    """, warehouse)
    assert retention.retention.iloc[0] > retention.retention.iloc[-1]


def test_user_metrics_are_one_row_per_assigned_user(warehouse, tables):
    counts = query("""
        select count(*) as rows, count(distinct user_id) as users
        from main_marts.fct_user_metrics
    """, warehouse)
    assert counts.rows.iloc[0] == counts.users.iloc[0] == len(tables["assignment"])


def test_the_pre_period_reaches_the_warehouse(warehouse):
    """Without it CUPED silently does nothing here while working in the validation study — the two
    halves of the project would be analysing different data."""
    periods = query("select period, count(*) as n from main_intermediate.int_user_period group by 1",
                    warehouse)
    assert set(periods.period) == {"pre", "during"}
    covariate = query("select avg(revenue_pre) as r, avg(sessions_pre) as s "
                      "from main_marts.fct_user_metrics", warehouse)
    assert covariate.r.iloc[0] > 0 and covariate.s.iloc[0] > 0


def test_cumulative_metrics_never_decrease(warehouse):
    daily = query("""
        select variant, session_date, cumulative_sessions
        from main_marts.mart_experiment_daily order by variant, session_date
    """, warehouse)
    for _, group in daily.groupby("variant"):
        assert group.cumulative_sessions.is_monotonic_increasing


@pytest.mark.parametrize("name", ["experiment_srm_check", "rfm_segments", "session_gaps",
                                  "funnel_by_segment"])
def test_the_standalone_queries_run(warehouse, name, monkeypatch):
    """These are the queries a reader is most likely to copy, so a broken one is worse than useless.
    DuckDB resolves the mart names, so they only need the built warehouse."""
    frame = run_sql_file(ROOT / "sql" / f"{name}.sql", warehouse)
    assert not frame.empty


def test_the_srm_check_passes_on_a_correctly_randomised_experiment(warehouse):
    """It must not cry wolf: the generator randomises correctly, so the check should say so."""
    frame = run_sql_file(ROOT / "sql" / "experiment_srm_check.sql", warehouse)
    assert (frame.chi_square < 10.828).all()
    assert frame.verdict.str.contains("as intended").all()


def test_rfm_revenue_shares_sum_to_one_hundred(warehouse):
    frame = run_sql_file(ROOT / "sql" / "rfm_segments.sql", warehouse)
    assert frame.share_of_revenue_pct.sum() == pytest.approx(100.0, abs=0.2)

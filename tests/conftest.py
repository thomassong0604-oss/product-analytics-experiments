"""Shared fixtures. Nothing here touches the network.

The warehouse fixture builds a real DuckDB database and runs the real dbt project against it, so the
mart tests check the SQL that ships rather than a Python reimplementation of it. It is small and
session-scoped, which keeps the whole suite to a few seconds.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pae.generate import Spec, dataset, simulate_units
from pae.warehouse import load

ROOT = Path(__file__).resolve().parents[1]
SMALL = Spec(n_users=1_200, days=35, true_lift=0.20, seed=20260101)


@pytest.fixture(scope="session")
def spec() -> Spec:
    return SMALL


@pytest.fixture(scope="session")
def tables(spec):
    return dataset(spec)


@pytest.fixture(scope="session")
def units(spec):
    return simulate_units(spec)


@pytest.fixture(scope="session")
def arms(units):
    """(control, treatment) frames, the shape every estimator takes."""
    return units[units.variant == "control"], units[units.variant == "treatment"]


@pytest.fixture(scope="session")
def warehouse(tmp_path_factory, tables):
    """A built warehouse: raw tables loaded, then `dbt build` run over them.

    Skips rather than fails if dbt is not installed, so the rest of the suite still runs for someone
    who only wants the statistics. CI installs it, so the models are always exercised there.
    """
    if shutil.which("dbt") is None:
        pytest.skip("dbt is not installed; mart tests need it")
    db = tmp_path_factory.mktemp("warehouse") / "test.duckdb"
    load(tables, db)
    environment = {
        **os.environ,
        "DBT_PROFILES_DIR": str(ROOT / "dbt"),
        "DBT_TARGET_PATH": str(db.parent / "dbt_target"),
        "DBT_LOG_PATH": str(db.parent / "dbt_logs"),
        # profiles.yml resolves its path relative to the dbt directory, so point it at this one.
        "PAE_DB_PATH": str(db),
    }
    profile = (ROOT / "dbt" / "profiles.yml").read_text().replace(
        'path: "../data/warehouse.duckdb"', f'path: "{db}"')
    temporary_profiles = db.parent / "profiles.yml"
    temporary_profiles.write_text(profile)
    environment["DBT_PROFILES_DIR"] = str(db.parent)
    result = subprocess.run(["dbt", "build", "--no-use-colors"], cwd=ROOT / "dbt",
                            env=environment, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        pytest.fail(f"dbt build failed:\n{result.stdout[-4000:]}\n{result.stderr[-2000:]}")
    return db

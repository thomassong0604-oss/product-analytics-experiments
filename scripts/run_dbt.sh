#!/usr/bin/env bash
# Build and test the warehouse. Runs from the dbt directory so profiles.yml is found without
# needing a ~/.dbt setup, which keeps the project self-contained and CI-friendly.
set -euo pipefail
cd "$(dirname "$0")/../dbt"
export DBT_PROFILES_DIR="${DBT_PROFILES_DIR:-$(pwd)}"
export DBT_TARGET_PATH="${DBT_TARGET_PATH:-../data/dbt_target}"
export DBT_LOG_PATH="${DBT_LOG_PATH:-../data/dbt_logs}"
dbt build --no-use-colors "$@"

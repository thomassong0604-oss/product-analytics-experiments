#!/usr/bin/env bash
# The whole pipeline: seed, transform, analyse, validate, publish. No network, no credentials.
#
#   ./scripts/run_all.sh              # full size, a few minutes
#   ./scripts/run_all.sh --quick      # small, for CI and a fast local check
set -euo pipefail
cd "$(dirname "$0")/.."

USERS=120000; LIFT=0.15; REPS=600
if [[ "${1:-}" == "--quick" ]]; then USERS=12000; REPS=60; fi

echo "== 1/5 seed  (${USERS} users, true lift ${LIFT}) =="
python3 scripts/seed.py --users "$USERS" --lift "$LIFT"

echo "== 2/5 transform and test (dbt) =="
./scripts/run_dbt.sh

echo "== 3/5 experiment readout =="
python3 scripts/analyze_experiment.py --out results

echo "== 4/5 validate the analysis against known truth (${REPS} replications) =="
python3 scripts/validate_methods.py --replications "$REPS" --users 6000 --lift "$LIFT" --out results

echo "== 5/5 dashboard =="
python3 dashboard/build_dashboard.py --out docs/index.html

echo
echo "done — results/validation.md, results/readout.json, docs/index.html"

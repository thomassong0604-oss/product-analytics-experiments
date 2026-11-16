#!/usr/bin/env python3
"""Read the warehouse and produce an experiment readout.

    python scripts/analyze_experiment.py --db data/warehouse.duckdb --out results

Reports every metric with the estimator that belongs to its grain, states the minimum detectable
effect beside the result so a null is not mistaken for a flat feature, and prints the naive
session-level standard error next to the clustered one so the difference is visible in the readout
itself rather than only in the methods appendix.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pae.experiment import (cuped, delta_method_ratio, min_detectable_effect, naive_ratio,  # noqa: E402
                            two_proportion, welch_mean)
from pae.experiment.power import design_effect                                              # noqa: E402
from pae.warehouse import DEFAULT_DB, query                                                 # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    units = query("select * from main_marts.fct_user_metrics", args.db)
    if units.empty:
        sys.exit("fct_user_metrics is empty — run scripts/seed.py then scripts/run_dbt.sh")
    control = units[units.variant == "control"]
    treatment = units[units.variant == "treatment"]

    results = {
        "experiment": str(units.experiment.iloc[0]),
        "users": {"control": len(control), "treatment": len(treatment)},
        "sessions": {"control": int(control.sessions.sum()),
                     "treatment": int(treatment.sessions.sum())},
    }

    readouts = {
        "purchases per session (clustered on user)":
            delta_method_ratio(control.purchases, control.sessions,
                               treatment.purchases, treatment.sessions),
        "purchases per session (sessions as independent — WRONG)":
            naive_ratio(control.purchases, control.sessions,
                        treatment.purchases, treatment.sessions),
        "users who converted":
            two_proportion(int(control.converted.sum()), len(control),
                           int(treatment.converted.sum()), len(treatment)),
        "revenue per user":
            welch_mean(control.revenue, treatment.revenue),
        "revenue per user (CUPED)":
            cuped(control.revenue, control.revenue_pre, treatment.revenue, treatment.revenue_pre),
        "sessions per user (negative control)":
            welch_mean(control.sessions, treatment.sessions),
    }

    print(f"\n{results['experiment']}: {len(control):,} control / {len(treatment):,} treatment "
          f"users, {results['sessions']['control'] + results['sessions']['treatment']:,} sessions\n")
    width = max(len(k) for k in readouts)
    for label, effect in readouts.items():
        print(f"  {label:<{width}}  {effect}")
        results[label] = {"relative": effect.relative, "rel_low": effect.rel_low,
                          "rel_high": effect.rel_high, "absolute": effect.absolute,
                          "standard_error": effect.standard_error, "p_value": effect.p_value,
                          "significant": effect.significant, "method": effect.method}

    clustered = readouts["purchases per session (clustered on user)"]
    naive = readouts["purchases per session (sessions as independent — WRONG)"]
    inflation = clustered.standard_error / naive.standard_error
    sessions_per_user = float(units.sessions.mean())
    results["se_inflation_from_clustering"] = inflation
    print(f"\n  Clustering widens the standard error by {inflation - 1:.1%} "
          f"({sessions_per_user:.1f} sessions per user). A readout using the naive figure would "
          f"call\n  an effect {inflation:.2f}x smaller than this one significant.")

    baseline = float(clustered.mean_control)
    mde = min_detectable_effect(baseline, len(control),
                               design=design_effect(sessions_per_user, 0.3))
    results["min_detectable_effect"] = mde
    print(f"\n  Smallest relative lift detectable at this size: {mde['relative']:.1%} "
          f"(80% power, design effect {mde['design_effect']:.2f}).")
    if not clustered.significant:
        print("  The headline metric did not separate. At this size that is the expected outcome\n"
              "  for any true effect below the figure above — it is not evidence of no effect.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "readout.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n-> {out / 'readout.json'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

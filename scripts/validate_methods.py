#!/usr/bin/env python3
"""Check the analysis against data whose true effect is known, and write the report.

    python scripts/validate_methods.py --replications 500 --users 6000 --out results

This is the part of the project that justifies the rest. Each study runs the same experiment many
times on fresh data and asks whether the estimators kept their promises: a 5% rejection rate when
nothing is happening, a 95% interval that contains the truth, and a point estimate that is not
systematically off. Nothing here is a claim about a product.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pae.generate import Spec                                      # noqa: E402
from pae.experiment.sequential import peeking_alpha, sequential_boundaries  # noqa: E402
from pae.experiment.validate import (attenuation, bias, coverage, false_positive_rate,  # noqa: E402
                                     negative_controls, power, replicate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replications", type=int, default=500)
    parser.add_argument("--users", type=int, default=6000)
    parser.add_argument("--lift", type=float, default=0.15)
    parser.add_argument("--looks", type=int, default=14)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"replications": args.replications, "users": args.users,
                                 "stated_lift": args.lift}
    lines: list[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append(f"### {title}")
        lines.append("")
        print(f"\n== {title}")

    print(f"A/A study: {args.replications} replications at {args.users:,} users")
    aa = replicate(Spec(n_users=args.users, true_lift=0.0), args.replications, progress=True)
    print(f"A/B study: {args.replications} replications, true lift {args.lift:+.0%}")
    ab = replicate(Spec(n_users=args.users, true_lift=args.lift), args.replications, progress=True)

    lines += ["# Validation report", "",
              f"{args.replications} replications per study, {args.users:,} users each. "
              f"Every rate below is itself an estimate, so it carries a Wilson interval; a verdict "
              f"is given only when the nominal value falls outside it.", ""]

    section("False positive rate, A/A (nominal 0.05)")
    lines += ["| estimator | metric | variance treated as | rejection rate | verdict |",
              "|---|---|---|---|---|"]
    checks = false_positive_rate(aa)
    report["false_positive_rate"] = [{"estimator": c.estimator, "rate": c.rate.point,
                                      "low": c.rate.low, "high": c.rate.high,
                                      "holds": c.holds} for c in checks]
    for c in checks:
        print("  ", c)
        lines.append(f"| `{c.estimator}` | {c.metric} | {c.variance} | {c.rate} | "
                     f"{'as promised' if c.holds else '**miscalibrated**'} |")

    section("Interval coverage, A/B (nominal 0.95)")
    lines += ["| estimator | metric | coverage | verdict |", "|---|---|---|---|"]
    checks = coverage(ab)
    report["coverage"] = [{"estimator": c.estimator, "rate": c.rate.point, "low": c.rate.low,
                           "high": c.rate.high, "holds": c.holds} for c in checks]
    for c in checks:
        print("  ", c)
        lines.append(f"| `{c.estimator}` | {c.metric} | {c.rate} | "
                     f"{'as promised' if c.holds else '**miscalibrated**'} |")

    section("Does each estimator's claimed precision match how much it actually moves?")
    frame = bias(aa)
    report["honesty"] = frame.to_dict("records")
    lines += ["An honesty ratio below 1 means the estimator reports a standard error smaller than "
              "the spread of its own estimates across replications — it is overstating its own "
              "precision.", "",
              "| estimator | claimed SE | actual spread | honesty ratio | unbiased |",
              "|---|---|---|---|---|"]
    for row in frame.sort_values("honesty_ratio").itertuples():
        lines.append(f"| `{row.estimator}` | {row.claimed_se:.4f} | {row.actual_spread:.4f} | "
                     f"**{row.honesty_ratio:.2f}** | {'yes' if row.unbiased else 'no'} |")
    print(frame[["estimator", "claimed_se", "actual_spread", "honesty_ratio",
                 "unbiased"]].round(4).to_string(index=False))

    section("Negative control: a metric the treatment cannot move")
    checks = negative_controls(ab)
    report["negative_controls"] = [{"estimator": c.estimator, "rate": c.rate.point,
                                    "holds": c.holds} for c in checks]
    lines += ["While a real effect exists elsewhere, an estimator run on sessions per user should "
              "still find nothing. If it does find something, the effect is leaking into a metric "
              "it has no mechanism to touch.", "",
              "| estimator | metric | rejection rate | verdict |", "|---|---|---|---|"]
    for c in checks:
        print("  ", c)
        lines.append(f"| `{c.estimator}` | {c.metric} | {c.rate} | "
                     f"{'as promised' if c.holds else '**leaking**'} |")

    section("How much the choice of metric changes the headline")
    frame = attenuation(ab)
    report["attenuation"] = frame.to_dict("records")
    lines += [f"The generator applied a {args.lift:+.0%} lift to the probability that a session "
              "ends in a purchase. Read on a different metric, the same effect reports a different "
              "number — and on a saturating one it reports a smaller one.", "",
              "| metric | observed lift | share of the stated lift |", "|---|---|---|"]
    for row in frame.itertuples():
        lines.append(f"| {row.metric} | {row.observed_lift:+.2%} "
                     f"[{row.observed_low:+.2%}, {row.observed_high:+.2%}] | "
                     f"{row.share_of_stated_lift:.2f} |")
    print(frame[["metric", "observed_lift", "share_of_stated_lift"]].round(4).to_string(index=False))

    section(f"What peeking costs, and what fixes it ({args.looks} looks)")
    peek = peeking_alpha(args.looks, iterations=40_000)
    report["peeking"] = peek
    report["boundaries"] = {
        "pocock": sequential_boundaries(args.looks, 0.05, "pocock"),
        "obrien_fleming": sequential_boundaries(args.looks, 0.05, "obrien-fleming"),
    }
    lines += ["Stopping at the first look where p < 0.05 does not give a 5% false positive rate. "
              "The boundaries are calibrated by exact numerical integration, so they hold their "
              "nominal rate rather than approximately holding it.", "",
              "| stopping rule | false positive rate |", "|---|---|",
              f"| stop at the first look where p < 0.05 | **{peek['naive_peeking']:.3f}** |",
              f"| one analysis at a fixed horizon | {peek['fixed_horizon']:.3f} |",
              f"| Pocock boundary | {peek['pocock']:.3f} |",
              f"| O'Brien-Fleming boundary | {peek['obrien_fleming']:.3f} |"]
    for key in ("naive_peeking", "fixed_horizon", "pocock", "obrien_fleming"):
        print(f"   {key:16s} {peek[key]:.4f}")

    section("Power to detect the stated lift")
    checks = power(ab)
    report["power"] = [{"estimator": c.estimator, "rate": c.rate.point} for c in checks]
    lines += ["| estimator | metric | power |", "|---|---|---|"]
    for c in checks:
        lines.append(f"| `{c.estimator}` | {c.metric} | {c.rate} |")

    (out / "validation.md").write_text("\n".join(lines) + "\n")
    (out / "validation.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n-> {out / 'validation.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

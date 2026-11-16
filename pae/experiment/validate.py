"""Does the analysis tell the truth? Measured, not assumed.

Every estimator in `pae.experiment.estimate` promises two things: that its interval contains the
true effect 95% of the time, and that when there is no effect it declares one 5% of the time. Those
promises are checkable, because `pae.generate` knows the true effect. This module checks them.

What it produces is not a result about the product. It is a result about the analysis — and it is
the reason the rest of the numbers in this repository can be read at face value. Two findings fall
out of it that matter for any real readout:

  the session-level standard error is too small     because a user's sessions are correlated, so
                                                    treating them as independent rejects the null
                                                    far more often than 5% of the time
  CUPED pays off in proportion to rho squared       so it is worth a lot on a persistent metric and
                                                    nearly nothing on a sparse one
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..generate import Spec, simulate_units
from ..stats import Rate, wilson
from .estimate import cuped, delta_method_ratio, naive_ratio, two_proportion, welch_mean

# estimator -> (metric it measures, how its variance is computed)
ESTIMATORS: dict[str, tuple[str, str]] = {
    "naive_ratio": ("purchases per session", "sessions treated as independent"),
    "delta_method_ratio": ("purchases per session", "clustered on user"),
    "two_proportion": ("users who converted", "one row per randomised unit"),
    "welch_mean": ("revenue per user", "unequal variances"),
    "cuped_revenue": ("revenue per user", "CUPED on pre-period revenue"),
    "cuped_sessions": ("sessions per user", "CUPED on pre-period sessions"),
}

# What the true lift *is* for each metric, as a multiple of the generator's `true_lift`.
#
# This mapping is the part that makes a coverage study mean anything, and getting it wrong is an
# easy way to accuse a correct estimator of being broken. The generator applies its lift to the
# probability that a session ends in a purchase. That does not make every metric move by the same
# amount:
#
#   purchases per session   exactly the stated lift. The treatment multiplies this probability and
#                           nothing else, and no probability is clipped, so the ratio of arm means
#                           is exactly (1 + lift).
#   revenue per user        also exactly the stated lift. Revenue per user is
#                           E[sessions] x P(purchase | session) x E[order value]; the treatment
#                           touches only the middle term, so the relative change carries through
#                           unchanged.
#   sessions per user       exactly zero. The treatment cannot affect how often someone visits, so
#                           this is a negative control: an estimator run on it should keep finding
#                           nothing even when a real effect exists elsewhere. A pipeline that lights
#                           this up has a bug, a leak, or broken randomisation.
#   users who converted     not a fixed multiple at all. "Purchased at least once" is
#                           1 - (1 - q)^n, which saturates, so a lift on q arrives attenuated and by
#                           an amount that depends on the session distribution. It is excluded from
#                           the coverage check for that reason and reported separately by
#                           `attenuation`, because the size of the attenuation is itself worth
#                           knowing: the same experiment looks weaker when read on this metric.
METRIC_TRUTH: dict[str, float | None] = {
    "purchases per session": 1.0,
    "revenue per user": 1.0,
    "sessions per user": 0.0,
    "users who converted": None,
}


def _effects(units: pd.DataFrame) -> dict[str, object]:
    """Every estimator applied to one simulated experiment."""
    control = units[units.variant == "control"]
    treatment = units[units.variant == "treatment"]
    return {
        "naive_ratio": naive_ratio(control.purchases, control.sessions,
                                   treatment.purchases, treatment.sessions),
        "delta_method_ratio": delta_method_ratio(control.purchases, control.sessions,
                                                 treatment.purchases, treatment.sessions),
        "two_proportion": two_proportion(int(control.converted.sum()), len(control),
                                         int(treatment.converted.sum()), len(treatment)),
        "welch_mean": welch_mean(control.revenue, treatment.revenue),
        "cuped_revenue": cuped(control.revenue, control.revenue_pre,
                               treatment.revenue, treatment.revenue_pre),
        "cuped_sessions": cuped(control.sessions, control.sessions_pre,
                                treatment.sessions, treatment.sessions_pre),
    }


def replicate(spec: Spec, replications: int = 500, progress: bool = False) -> pd.DataFrame:
    """Run the same experiment many times on fresh data, keeping what every estimator concluded.

    Only the seed changes between replications, so the spread across rows is sampling variation and
    nothing else — which is the only way a rejection rate means anything.
    """
    rows = []
    for i in range(replications):
        units = simulate_units(Spec(**{**spec.__dict__, "seed": spec.seed + i}))
        for name, effect in _effects(units).items():
            rows.append({
                "replication": i,
                "estimator": name,
                "metric": ESTIMATORS[name][0],
                "variance": ESTIMATORS[name][1],
                "relative": effect.relative,
                "rel_low": effect.rel_low,
                "rel_high": effect.rel_high,
                "absolute": effect.absolute,
                "standard_error": effect.standard_error,
                # The same uncertainty on the relative scale, recovered from the interval width.
                # Keeping it lets the study compare what an estimator *claimed* about its precision
                # with how much its estimate actually moved from one replication to the next, which
                # is the single most direct test of a standard error.
                "standard_error_relative": (effect.rel_high - effect.rel_low) / (2 * 1.96),
                "p_value": effect.p_value,
                "significant": effect.significant,
                "covers_true_lift": effect.covers_relative(
                    spec.true_lift * (METRIC_TRUTH[ESTIMATORS[name][0]] or 0.0)),
                "truth_known": METRIC_TRUTH[ESTIMATORS[name][0]] is not None,
                "true_lift_for_metric": (spec.true_lift * METRIC_TRUTH[ESTIMATORS[name][0]]
                                         if METRIC_TRUTH[ESTIMATORS[name][0]] is not None
                                         else float("nan")),
            })
        if progress and (i + 1) % 25 == 0:
            print(f"    {i + 1}/{replications}", flush=True)
    frame = pd.DataFrame(rows)
    frame.attrs["true_lift"] = spec.true_lift
    frame.attrs["n_users"] = spec.n_users
    return frame


@dataclass(frozen=True)
class Check:
    """One estimator's measured rate against the rate it promised."""
    estimator: str
    metric: str
    variance: str
    what: str
    nominal: float
    rate: Rate
    mean_standard_error: float

    @property
    def holds(self) -> bool:
        """True when the interval around the measured rate contains the nominal one.

        A verdict is withheld rather than guessed: with 500 replications the interval around a
        measured 0.05 is roughly ±0.02, so a measured 0.065 is not evidence of a broken estimator
        and this returns True for it. The study has to be large enough to say otherwise.
        """
        return self.rate.contains(self.nominal)

    def __str__(self) -> str:
        verdict = "as promised" if self.holds else "MISCALIBRATED"
        return (f"{self.estimator:22s} {self.what:18s} {self.rate} vs nominal {self.nominal:.2f}"
                f"  mean SE {self.mean_standard_error:.5f}  — {verdict}")


def false_positive_rate(frame: pd.DataFrame, alpha: float = 0.05) -> list[Check]:
    """How often each estimator declared an effect when the true lift was zero.

    Only meaningful on an A/A study. Running it against a frame with a real effect would measure
    power and call it a false positive rate, so that is refused rather than silently reported.
    """
    true_lift = frame.attrs.get("true_lift", None)
    if true_lift:
        raise ValueError(f"false positive rate needs an A/A study (true_lift=0), got {true_lift}")
    out = []
    for name, group in frame.groupby("estimator", sort=False):
        out.append(Check(name, group.metric.iloc[0], group.variance.iloc[0],
                         "rejection rate", alpha,
                         wilson(int(group.significant.sum()), len(group)),
                         float(group.standard_error.mean())))
    return sorted(out, key=lambda c: -c.rate.point)


def coverage(frame: pd.DataFrame, conf: float = 0.95) -> list[Check]:
    """How often each estimator's interval contained the true lift *for its own metric*.

    Estimators whose metric has no fixed true lift are left out rather than judged against a number
    that does not apply to them — see METRIC_TRUTH. `attenuation` reports those instead.
    """
    out = []
    for name, group in frame[frame.truth_known].groupby("estimator", sort=False):
        out.append(Check(name, group.metric.iloc[0], group.variance.iloc[0],
                         "interval coverage", conf,
                         wilson(int(group.covers_true_lift.sum()), len(group)),
                         float(group.standard_error.mean())))
    return sorted(out, key=lambda c: c.rate.point)


def negative_controls(frame: pd.DataFrame, alpha: float = 0.05) -> list[Check]:
    """Estimators run on a metric the treatment cannot move, while a real effect exists elsewhere.

    These should keep rejecting at the nominal rate. If they do not, the effect is leaking into a
    metric it has no mechanism to touch, and the pipeline — not the feature — is what moved.
    """
    out = []
    controls = frame[frame.true_lift_for_metric == 0.0]
    for name, group in controls.groupby("estimator", sort=False):
        out.append(Check(name, group.metric.iloc[0], group.variance.iloc[0],
                         "rejection rate", alpha,
                         wilson(int(group.significant.sum()), len(group)),
                         float(group.standard_error.mean())))
    return sorted(out, key=lambda c: -c.rate.point)


def attenuation(frame: pd.DataFrame) -> pd.DataFrame:
    """How much each metric shrinks the stated lift.

    A saturating metric reads a real improvement as a smaller one: "purchased at least once" cannot
    rise much among users who already purchase, so a lift applied to every purchase decision shows
    up attenuated. This table is how much — which is to say, how much the choice of metric alone
    changes the headline number in a readout.
    """
    stated = frame.attrs.get("true_lift", 0.0)
    rows = []
    for metric, group in frame.groupby("metric", sort=False):
        mean = float(group.relative.mean())
        spread = float(group.relative.std(ddof=1)) / len(group) ** 0.5
        rows.append({
            "metric": metric,
            "stated_lift_on_purchase_step": stated,
            "observed_lift": mean,
            "observed_low": mean - 1.96 * spread,
            "observed_high": mean + 1.96 * spread,
            "share_of_stated_lift": mean / stated if stated else float("nan"),
            "true_lift_known": bool(group.truth_known.iloc[0]),
            "replications": len(group),
        })
    return pd.DataFrame(rows)


def power(frame: pd.DataFrame) -> list[Check]:
    """How often each estimator detected a real effect. Nominal power is not fixed, so the nominal
    column is set to the conventional 0.80 target and the verdict is read with that in mind."""
    out = []
    for name, group in frame.groupby("estimator", sort=False):
        out.append(Check(name, group.metric.iloc[0], group.variance.iloc[0],
                         "power", 0.80,
                         wilson(int(group.significant.sum()), len(group)),
                         float(group.standard_error.mean())))
    return sorted(out, key=lambda c: -c.rate.point)


def bias(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean estimated lift against the true one — the check that the point estimate is unbiased.

    Separated from coverage on purpose: an estimator can be unbiased and still report intervals that
    are far too narrow, which is precisely the naive session-level case.
    """
    true_lift = frame.attrs.get("true_lift", 0.0)
    rows = []
    for name, group in frame.groupby("estimator", sort=False):
        mean = float(group.relative.mean())
        spread = float(group.relative.std(ddof=1))
        standard_error = spread / len(group) ** 0.5
        claimed = float(group.standard_error_relative.mean())
        rows.append({
            "estimator": name, "metric": group.metric.iloc[0],
            "true_lift": true_lift, "mean_estimate": mean,
            "bias": mean - true_lift,
            "bias_low": mean - true_lift - 1.96 * standard_error,
            "bias_high": mean - true_lift + 1.96 * standard_error,
            "actual_spread": spread,
            "claimed_se": claimed,
            # Below 1 the estimator is overstating its own precision. This is the clearest single
            # number in the study: it needs no distributional assumption, only the observation that
            # an honest standard error should match how much the estimate actually moves.
            "honesty_ratio": claimed / spread if spread > 0 else float("nan"),
            "replications": len(group),
        })
    out = pd.DataFrame(rows)
    # Unbiased here means the interval around the measured bias contains zero.
    out["unbiased"] = (out.bias_low <= 0) & (out.bias_high >= 0)
    return out

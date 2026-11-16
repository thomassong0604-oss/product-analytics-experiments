"""What the experiment could ever have detected.

Reported before the readout, not after. A result that was never detectable at this sample size is
not evidence of no effect, and a "flat" test on 2,000 users usually means the test was too small
rather than that the feature did nothing. The design effect is included because almost every
product metric is clustered, and ignoring that overstates the achievable precision — often by a
factor of two or more, which is the difference between a two-week test and a two-month one.
"""
from __future__ import annotations

import math

from scipy import stats


def _z(level: float) -> float:
    """The standard normal quantile at `level`.

    From scipy rather than a hand-rolled rational approximation. An approximation accurate in the
    middle of the distribution and loose in the tail is exactly the wrong trade here: every value
    this function is asked for is a tail quantile, and a boundary that is off by 0.1 in z turns a
    5% error rate into something else entirely. An earlier version of this file made that mistake
    and the sequential boundaries it produced did not hold their nominal rate.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must be strictly between 0 and 1")
    return float(stats.norm.ppf(level))


def design_effect(sessions_per_unit_mean: float, intra_unit_correlation: float) -> float:
    """How much clustering inflates the variance: 1 + (m̄ − 1)·ρ.

    With 2.6 sessions per user and a within-user correlation of 0.3, the variance is 1.48 times what
    an independence assumption gives — so the usable sample size is two thirds of the session count,
    not all of it.
    """
    return 1.0 + max(sessions_per_unit_mean - 1.0, 0.0) * max(intra_unit_correlation, 0.0)


def sample_size_for_proportions(baseline: float, relative_lift: float, power: float = 0.80,
                                alpha: float = 0.05, allocation: float = 0.5,
                                design: float = 1.0) -> dict:
    """Units per arm needed to detect a relative lift on a proportion.

    `design` multiplies the requirement to account for clustering, and `allocation` allows an uneven
    split — an imbalanced test needs more total units than a 50/50 one for the same power, which is
    easy to forget when a rollout starts at 10%.
    """
    if not 0 < baseline < 1:
        raise ValueError("baseline must be a proportion")
    if relative_lift == 0:
        raise ValueError("a zero lift needs an infinite sample")
    p1 = baseline
    p2 = baseline * (1 + relative_lift)
    if not 0 < p2 < 1:
        raise ValueError(f"lift takes the rate to {p2:.3f}, which is not a proportion")
    z_alpha, z_beta = _z(1 - alpha / 2), _z(power)
    p_bar = allocation * p2 + (1 - allocation) * p1
    numerator = (z_alpha * math.sqrt(p_bar * (1 - p_bar) / (allocation * (1 - allocation)))
                 + z_beta * math.sqrt(p1 * (1 - p1) / (1 - allocation)
                                      + p2 * (1 - p2) / allocation)) ** 2
    per_unit = numerator / (p2 - p1) ** 2 * design
    n_control = math.ceil(per_unit * (1 - allocation))
    n_treatment = math.ceil(per_unit * allocation)
    return {"baseline": p1, "target": p2, "absolute_effect": p2 - p1,
            "relative_lift": relative_lift, "power": power, "alpha": alpha,
            "design_effect": design, "n_control": n_control, "n_treatment": n_treatment,
            "n_total": n_control + n_treatment}


def min_detectable_effect(baseline: float, n_per_arm: int, power: float = 0.80,
                          alpha: float = 0.05, design: float = 1.0) -> dict:
    """The smallest relative lift this sample size could detect — the number to publish beside a
    null result, so "no effect" is not confused with "no power".

    Solved as the exact inverse of `sample_size_for_proportions` rather than from the simpler
    closed form. The two differ by which variance goes into the alpha term: the sample-size formula
    pools under the null, the textbook detectable-effect formula does not. Both are defensible, but
    two functions in the same module answering the same question in different ways is how a planning
    number and a readout number end up quietly disagreeing. Inverting the one that already exists
    makes them consistent by construction.
    """
    if n_per_arm <= 0:
        raise ValueError("need a positive sample size")
    if not 0 < baseline < 1:
        raise ValueError("baseline must be a proportion")

    def needed(lift: float) -> float:
        return sample_size_for_proportions(baseline, lift, power, alpha,
                                           design=design)["n_control"]

    low, high = 1e-6, (1.0 - baseline) / baseline * 0.999      # the largest lift still a proportion
    if needed(high) > n_per_arm:
        return {"baseline": baseline, "n_per_arm": n_per_arm, "power": power, "alpha": alpha,
                "design_effect": design, "absolute": float("nan"), "relative": float("nan"),
                "note": "no lift is detectable at this sample size"}
    for _ in range(80):
        mid = (low + high) / 2
        if needed(mid) > n_per_arm:
            low = mid                       # too small an effect to see at this n
        else:
            high = mid
    relative = (low + high) / 2
    return {"baseline": baseline, "n_per_arm": n_per_arm, "power": power, "alpha": alpha,
            "design_effect": design, "absolute": baseline * relative, "relative": relative}

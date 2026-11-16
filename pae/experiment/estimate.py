"""Treatment effect estimators, each with the variance that actually belongs to it.

The estimators here differ almost entirely in how they compute a standard error, and that is the
point. Getting a point estimate right is easy — both arms' means are a `GROUP BY`. Getting the
uncertainty right is where experiment readouts go wrong, and the two ways they go wrong have their
own functions so the difference can be measured rather than argued about:

  `naive_ratio`         treats every session as an independent observation. It is what a SQL query
                        returns by default, and when users have several sessions it is too narrow —
                        so a readout built on it rejects the null far more often than 5% of the time.
  `delta_method_ratio`  treats the user as the unit, which is the unit randomisation actually used.

Both are kept because `pae.experiment.validate` runs them side by side and measures the false
positive rate of each. A claim that clustering matters is worth more with a number attached.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


@dataclass(frozen=True)
class Effect:
    """An absolute difference and the relative lift, both with an interval."""
    mean_control: float
    mean_treatment: float
    absolute: float
    abs_low: float
    abs_high: float
    relative: float
    rel_low: float
    rel_high: float
    standard_error: float
    p_value: float
    n_control: int
    n_treatment: int
    unit: str
    method: str

    @property
    def significant(self) -> bool:
        """True when the interval for the absolute difference excludes zero."""
        return (self.abs_low > 0) or (self.abs_high < 0)

    def covers(self, true_absolute: float) -> bool:
        """Whether the absolute interval contains a known true effect."""
        return self.abs_low <= true_absolute <= self.abs_high

    def covers_relative(self, true_relative: float) -> bool:
        """Whether the relative interval contains a known true lift.

        This is the check the coverage study uses, because the generator's input is a relative lift
        and comparing against it needs no knowledge of the population base rate.
        """
        return self.rel_low <= true_relative <= self.rel_high

    def __str__(self) -> str:
        verdict = "significant" if self.significant else "not significant"
        return (f"{self.relative:+.2%} [{self.rel_low:+.2%}, {self.rel_high:+.2%}] "
                f"(abs {self.absolute:+.5f}, SE {self.standard_error:.5f}, p={self.p_value:.4f}, "
                f"{verdict}, by {self.unit}, {self.method})")


def _normal_p(z: float) -> float:
    """Two-sided p-value from a z statistic, via the complementary error function."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def _package(mean_c: float, mean_t: float, se_c: float, se_t: float, n_c: int, n_t: int,
             unit: str, method: str, conf: float = 0.95) -> Effect:
    """Assemble an Effect, with each arm's standard error kept separate.

    The two are needed apart because the relative interval is not the absolute interval divided by
    the control mean. That shortcut treats the control mean as a known constant and so reports an
    interval narrower than the truth — which would show up as under-coverage in the replication
    study and be mistaken for a broken estimator.

    Instead the relative effect is taken on the log scale, where the two arms separate:

        Var(log(μ_t / μ_c)) ≈ SE_t² / μ_t² + SE_c² / μ_c²

    and the interval is exponentiated back. Both means must be positive for that to be defined; if
    either is not, the relative interval is reported as undefined rather than as a number.
    """
    z = Z.get(conf, 1.96)
    se = math.sqrt(se_c ** 2 + se_t ** 2)
    diff = mean_t - mean_c
    low, high = diff - z * se, diff + z * se
    if mean_c > 0 and mean_t > 0:
        log_se = math.sqrt((se_t / mean_t) ** 2 + (se_c / mean_c) ** 2)
        ratio = mean_t / mean_c
        rel = ratio - 1.0
        rel_low = math.exp(math.log(ratio) - z * log_se) - 1.0
        rel_high = math.exp(math.log(ratio) + z * log_se) - 1.0
    else:
        rel = rel_low = rel_high = float("nan")
    p = _normal_p(diff / se) if se > 0 else 1.0
    return Effect(mean_c, mean_t, diff, low, high, rel, rel_low, rel_high, se, p, n_c, n_t,
                  unit, method)


# --------------------------------------------------------------------------- per-unit metrics

def two_proportion(successes_c: int, n_c: int, successes_t: int, n_t: int,
                   conf: float = 0.95) -> Effect:
    """Difference of two proportions where each observation is one randomised unit.

    Correct only when the unit counted here is the unit randomised — one row per user for a
    user-randomised experiment. Counting sessions here is the mistake `naive_ratio` exists to
    demonstrate.
    """
    if n_c <= 0 or n_t <= 0:
        raise ValueError("both arms need observations")
    p_c, p_t = successes_c / n_c, successes_t / n_t
    return _package(p_c, p_t, math.sqrt(p_c * (1 - p_c) / n_c), math.sqrt(p_t * (1 - p_t) / n_t),
                    n_c, n_t, "unit", "two-proportion z", conf)


def welch_mean(control: np.ndarray, treatment: np.ndarray, conf: float = 0.95) -> Effect:
    """Difference of means with unequal variances — the right default for revenue per user.

    Welch rather than Student because an experiment that changes behaviour usually changes spread as
    well, and revenue is heavy-tailed enough that assuming equal variances is not a small sin.
    """
    control, treatment = np.asarray(control, float), np.asarray(treatment, float)
    if control.size < 2 or treatment.size < 2:
        raise ValueError("both arms need at least two observations")
    return _package(control.mean(), treatment.mean(),
                    math.sqrt(control.var(ddof=1) / control.size),
                    math.sqrt(treatment.var(ddof=1) / treatment.size),
                    control.size, treatment.size, "unit", "Welch", conf)


# --------------------------------------------------------------------------- ratio metrics

def naive_ratio(y_control: np.ndarray, n_control: np.ndarray,
                y_treatment: np.ndarray, n_treatment: np.ndarray, conf: float = 0.95) -> Effect:
    """Purchases per session, pretending every session is independent.

    This is what `SUM(purchases) / SUM(sessions)` with a binomial standard error gives you, and it
    is wrong whenever one user contributes several sessions, because those sessions share the user's
    own propensity. It is implemented so the error can be quantified, not because it should be used.
    """
    sum_y_c, sum_n_c = float(np.sum(y_control)), float(np.sum(n_control))
    sum_y_t, sum_n_t = float(np.sum(y_treatment)), float(np.sum(n_treatment))
    if sum_n_c <= 0 or sum_n_t <= 0:
        raise ValueError("both arms need denominator events")
    p_c, p_t = sum_y_c / sum_n_c, sum_y_t / sum_n_t
    return _package(p_c, p_t, math.sqrt(p_c * (1 - p_c) / sum_n_c),
                    math.sqrt(p_t * (1 - p_t) / sum_n_t), int(sum_n_c), int(sum_n_t),
                    "session (wrong)", "binomial on sessions", conf)


def _ratio_variance(y: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """Ratio of sums and its delta-method variance, clustered on the unit.

    For R = Σy / Σn over k units,

        Var(R) ≈ ( Var(y) - 2·R·Cov(y, n) + R²·Var(n) ) / ( k · n̄² )

    which is the first-order expansion of the ratio around the two means. The covariance term is the
    one that matters in practice: heavy users have both more sessions and more purchases, so y and n
    move together and ignoring it misstates the variance in a direction you cannot predict.
    """
    y, n = np.asarray(y, float), np.asarray(n, float)
    k = y.size
    if k < 2:
        raise ValueError("need at least two units")
    n_bar = n.mean()
    if n_bar <= 0:
        raise ValueError("denominator is empty")
    ratio = y.sum() / n.sum()
    var_y, var_n = y.var(ddof=1), n.var(ddof=1)
    cov = float(np.cov(y, n, ddof=1)[0, 1])
    variance = (var_y - 2 * ratio * cov + ratio ** 2 * var_n) / (k * n_bar ** 2)
    return ratio, max(variance, 0.0)


def delta_method_ratio(y_control: np.ndarray, n_control: np.ndarray,
                       y_treatment: np.ndarray, n_treatment: np.ndarray,
                       conf: float = 0.95) -> Effect:
    """Purchases per session with the user as the unit of analysis.

    Same point estimate as the naive version — the difference is entirely in the standard error, and
    that difference is what keeps the false positive rate at its nominal level.
    """
    ratio_c, var_c = _ratio_variance(y_control, n_control)
    ratio_t, var_t = _ratio_variance(y_treatment, n_treatment)
    return _package(ratio_c, ratio_t, math.sqrt(var_c), math.sqrt(var_t),
                    len(y_control), len(y_treatment), "user",
                    "delta method, clustered on user", conf)


@dataclass(frozen=True)
class ClusterResult:
    ratio_control: float
    ratio_treatment: float
    absolute: float
    low: float
    high: float
    iterations: int


def cluster_bootstrap(y_control: np.ndarray, n_control: np.ndarray,
                      y_treatment: np.ndarray, n_treatment: np.ndarray,
                      iterations: int = 2000, conf: float = 0.95,
                      seed: int = 20260101) -> ClusterResult:
    """Resample users, not sessions — an assumption-light check on the delta method.

    Resampling sessions would destroy the within-user correlation and reproduce the naive interval,
    so the unit being resampled is the whole point. Agreement between this and the delta method is
    evidence that the delta-method approximation holds at this sample size.
    """
    rng = np.random.default_rng(seed)
    y_c, n_c = np.asarray(y_control, float), np.asarray(n_control, float)
    y_t, n_t = np.asarray(y_treatment, float), np.asarray(n_treatment, float)
    diffs = np.empty(iterations)
    for i in range(iterations):
        ic = rng.integers(0, y_c.size, y_c.size)
        it = rng.integers(0, y_t.size, y_t.size)
        denom_c, denom_t = n_c[ic].sum(), n_t[it].sum()
        diffs[i] = ((y_t[it].sum() / denom_t if denom_t else 0.0)
                    - (y_c[ic].sum() / denom_c if denom_c else 0.0))
    diffs.sort()
    alpha = 1 - conf
    return ClusterResult(y_c.sum() / n_c.sum(), y_t.sum() / n_t.sum(),
                         y_t.sum() / n_t.sum() - y_c.sum() / n_c.sum(),
                         float(diffs[int(alpha / 2 * iterations)]),
                         float(diffs[min(int((1 - alpha / 2) * iterations), iterations - 1)]),
                         iterations)


# --------------------------------------------------------------------------- variance reduction

def cuped(y_control: np.ndarray, x_control: np.ndarray,
          y_treatment: np.ndarray, x_treatment: np.ndarray, conf: float = 0.95) -> Effect:
    """CUPED: subtract off the part of the outcome a pre-experiment covariate already explained.

    Y' = Y − θ(X − E[X]), with θ the pooled regression coefficient of Y on X. Because X is measured
    *before* assignment it cannot have been affected by the treatment, so the adjustment leaves the
    estimate unbiased while removing the share of variance that was never about the experiment. The
    reduction is roughly ρ², so a covariate correlated 0.5 with the outcome buys about 25%.

    Taking θ from the pooled data rather than per arm matters: fitting it separately lets the
    treatment influence its own adjustment and reintroduces bias.
    """
    y_c, x_c = np.asarray(y_control, float), np.asarray(x_control, float)
    y_t, x_t = np.asarray(y_treatment, float), np.asarray(x_treatment, float)
    y_all, x_all = np.concatenate([y_c, y_t]), np.concatenate([x_c, x_t])
    var_x = x_all.var(ddof=1)
    theta = (float(np.cov(y_all, x_all, ddof=1)[0, 1]) / var_x) if var_x > 0 else 0.0
    grand_x = x_all.mean()
    adj_c = y_c - theta * (x_c - grand_x)
    adj_t = y_t - theta * (x_t - grand_x)
    effect = _package(adj_c.mean(), adj_t.mean(),
                      math.sqrt(adj_c.var(ddof=1) / adj_c.size),
                      math.sqrt(adj_t.var(ddof=1) / adj_t.size),
                      adj_c.size, adj_t.size, "unit", f"CUPED (theta={theta:.4f})", conf)
    return effect


def variance_reduction(y_control: np.ndarray, x_control: np.ndarray,
                       y_treatment: np.ndarray, x_treatment: np.ndarray) -> float:
    """Share of standard error CUPED removes here — the honest way to report its benefit."""
    plain = welch_mean(y_control, y_treatment)
    adjusted = cuped(y_control, x_control, y_treatment, x_treatment)
    if plain.standard_error == 0:
        return 0.0
    return 1.0 - adjusted.standard_error / plain.standard_error

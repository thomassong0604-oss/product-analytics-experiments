"""Estimators: known values, and the properties that distinguish them from each other."""
import numpy as np
import pytest

from pae.experiment.estimate import (cluster_bootstrap, cuped, delta_method_ratio, naive_ratio,
                                     two_proportion, variance_reduction, welch_mean)


def test_two_proportion_against_a_hand_computed_value():
    effect = two_proportion(100, 1000, 120, 1000)
    assert effect.mean_control == pytest.approx(0.10)
    assert effect.mean_treatment == pytest.approx(0.12)
    assert effect.absolute == pytest.approx(0.02)
    # SE = sqrt(.1*.9/1000 + .12*.88/1000) = sqrt(0.0001956)
    assert effect.standard_error == pytest.approx(0.0139857, rel=1e-4)
    assert effect.relative == pytest.approx(0.20)


def test_both_arms_are_required():
    with pytest.raises(ValueError):
        two_proportion(0, 0, 5, 10)
    with pytest.raises(ValueError):
        welch_mean(np.array([1.0]), np.array([1.0, 2.0]))


def test_the_relative_interval_is_wider_than_dividing_by_the_control_mean():
    """The naive shortcut treats the control mean as a known constant. The log-ratio interval does
    not, so it must be at least as wide — this is what keeps coverage at 95%."""
    effect = two_proportion(100, 1000, 120, 1000)
    shortcut_half_width = 1.96 * effect.standard_error / effect.mean_control
    proper_half_width = (effect.rel_high - effect.rel_low) / 2
    assert proper_half_width > shortcut_half_width


def test_naive_and_clustered_agree_on_the_point_estimate():
    """They differ only in the standard error. If the point estimates diverged, the comparison
    between them would be measuring two things at once."""
    rng = np.random.default_rng(3)
    n_c, n_t = 1 + rng.poisson(5, 800), 1 + rng.poisson(5, 800)
    y_c, y_t = rng.binomial(n_c, 0.1), rng.binomial(n_t, 0.12)
    naive = naive_ratio(y_c, n_c, y_t, n_t)
    clustered = delta_method_ratio(y_c, n_c, y_t, n_t)
    assert naive.absolute == pytest.approx(clustered.absolute)
    assert naive.relative == pytest.approx(clustered.relative)


def test_clustering_widens_the_interval_when_users_have_many_sessions():
    """The headline finding of the project, as a unit test: correlated sessions inside a user mean
    the session-level standard error is too small."""
    rng = np.random.default_rng(11)
    k = 1_500
    propensity_c, propensity_t = rng.beta(2, 18, k), rng.beta(2, 18, k)
    n_c, n_t = 1 + rng.poisson(8, k), 1 + rng.poisson(8, k)
    y_c, y_t = rng.binomial(n_c, propensity_c), rng.binomial(n_t, propensity_t)
    assert (delta_method_ratio(y_c, n_c, y_t, n_t).standard_error
            > naive_ratio(y_c, n_c, y_t, n_t).standard_error)


def test_with_one_session_each_the_two_variances_nearly_agree():
    """With no clustering there is nothing for the delta method to correct, so the two should come
    together. This is what shows the difference above is clustering and not a coding error."""
    rng = np.random.default_rng(5)
    k = 4_000
    n_c, n_t = np.ones(k, int), np.ones(k, int)
    y_c, y_t = rng.binomial(1, 0.1, k), rng.binomial(1, 0.1, k)
    naive = naive_ratio(y_c, n_c, y_t, n_t)
    clustered = delta_method_ratio(y_c, n_c, y_t, n_t)
    assert clustered.standard_error == pytest.approx(naive.standard_error, rel=0.02)


def test_the_cluster_bootstrap_agrees_with_the_delta_method():
    """Two different routes to the same interval. Agreement is evidence the delta-method
    approximation holds at this sample size; disagreement would mean it does not."""
    rng = np.random.default_rng(17)
    k = 1_200
    n_c, n_t = 1 + rng.poisson(6, k), 1 + rng.poisson(6, k)
    y_c = rng.binomial(n_c, rng.beta(2, 18, k))
    y_t = rng.binomial(n_t, rng.beta(2, 18, k))
    delta = delta_method_ratio(y_c, n_c, y_t, n_t)
    boot = cluster_bootstrap(y_c, n_c, y_t, n_t, iterations=1_500)
    assert boot.absolute == pytest.approx(delta.absolute)
    width_delta = delta.abs_high - delta.abs_low
    width_boot = boot.high - boot.low
    assert width_boot == pytest.approx(width_delta, rel=0.2)


def test_cuped_leaves_the_point_estimate_alone_when_the_covariate_is_noise():
    """An uninformative covariate should neither help nor bias. If it shifted the estimate, theta
    would be picking up something it should not."""
    rng = np.random.default_rng(23)
    y_c, y_t = rng.normal(10, 3, 3_000), rng.normal(10.5, 3, 3_000)
    x_c, x_t = rng.normal(0, 1, 3_000), rng.normal(0, 1, 3_000)
    plain = welch_mean(y_c, y_t)
    adjusted = cuped(y_c, x_c, y_t, x_t)
    assert adjusted.absolute == pytest.approx(plain.absolute, abs=0.1)
    assert adjusted.standard_error == pytest.approx(plain.standard_error, rel=0.05)


def test_cuped_reduces_variance_in_proportion_to_the_correlation():
    """The theory says the standard error falls by about 1 - sqrt(1 - rho^2). Checked against two
    correlations so the relationship, not just one number, is tested."""
    rng = np.random.default_rng(29)
    n = 6_000
    for rho in (0.5, 0.8):
        x_c, x_t = rng.normal(0, 1, n), rng.normal(0, 1, n)
        noise_c, noise_t = rng.normal(0, 1, n), rng.normal(0, 1, n)
        y_c = rho * x_c + (1 - rho ** 2) ** 0.5 * noise_c
        y_t = rho * x_t + (1 - rho ** 2) ** 0.5 * noise_t + 0.05
        expected = 1 - (1 - rho ** 2) ** 0.5
        assert variance_reduction(y_c, x_c, y_t, x_t) == pytest.approx(expected, abs=0.05)


def test_cuped_takes_theta_from_the_pooled_data():
    """Fitting theta per arm lets the treatment influence its own adjustment, which reintroduces the
    bias CUPED is supposed to avoid. A pooled theta is symmetric in the two arms."""
    rng = np.random.default_rng(31)
    n = 4_000
    x_c, x_t = rng.normal(0, 1, n), rng.normal(0, 1, n)
    y_c = 2 * x_c + rng.normal(0, 1, n)
    y_t = 2 * x_t + rng.normal(0, 1, n) + 0.3
    forward = cuped(y_c, x_c, y_t, x_t)
    backward = cuped(y_t, x_t, y_c, x_c)
    assert forward.absolute == pytest.approx(-backward.absolute, abs=1e-9)


def test_significance_is_read_from_the_interval_not_the_p_value():
    effect = two_proportion(500, 5_000, 700, 5_000)
    assert effect.significant and effect.abs_low > 0
    flat = two_proportion(500, 5_000, 505, 5_000)
    assert not flat.significant and flat.abs_low < 0 < flat.abs_high


def test_coverage_helpers_do_what_they_say():
    effect = two_proportion(100, 1_000, 120, 1_000)
    assert effect.covers(0.02) and not effect.covers(0.5)
    assert effect.covers_relative(0.20) and not effect.covers_relative(2.0)

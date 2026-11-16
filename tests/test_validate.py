"""The replication study. These tests check that the study would actually catch a bad estimator."""
import pytest

from pae.generate import Spec
from pae.experiment.validate import (METRIC_TRUTH, attenuation, bias, coverage,
                                     false_positive_rate, negative_controls, replicate)

REPLICATIONS = 60          # enough to detect a grossly miscalibrated estimator, fast enough for CI


@pytest.fixture(scope="module")
def aa():
    return replicate(Spec(n_users=2_500, true_lift=0.0), REPLICATIONS)


@pytest.fixture(scope="module")
def ab():
    return replicate(Spec(n_users=2_500, true_lift=0.30), REPLICATIONS)


def test_the_study_catches_the_estimator_that_understates_its_uncertainty(aa):
    """The headline finding, as a test. The session-level standard error is too small, so its
    rejection rate must come out above nominal — and the clustered one must not."""
    rates = {c.estimator: c for c in false_positive_rate(aa)}
    assert rates["naive_ratio"].rate.point > rates["delta_method_ratio"].rate.point
    assert rates["naive_ratio"].rate.low > 0.05, "the naive estimator should be caught"
    assert rates["delta_method_ratio"].holds


def test_the_honesty_ratio_separates_the_two(aa):
    frame = bias(aa).set_index("estimator")
    assert frame.loc["naive_ratio", "honesty_ratio"] < 0.9
    assert frame.loc["delta_method_ratio", "honesty_ratio"] == pytest.approx(1.0, abs=0.12)


def test_every_estimator_is_unbiased(aa):
    """An estimator can be unbiased and still report nonsense intervals, which is why bias and
    coverage are separate checks. Here all of them should pass the easy one."""
    assert bias(aa).unbiased.all()


def test_coverage_is_only_computed_where_the_truth_is_known(ab):
    """Judging an estimator against a lift that does not apply to its metric is how a correct
    estimator gets accused of being broken."""
    judged = {c.estimator for c in coverage(ab)}
    assert "two_proportion" not in judged, "per-user conversion saturates; its truth is not the input"
    assert {"delta_method_ratio", "welch_mean", "cuped_revenue"} <= judged


def test_the_clustered_estimator_covers_the_truth_and_the_naive_one_does_not(ab):
    rates = {c.estimator: c for c in coverage(ab)}
    assert rates["delta_method_ratio"].holds
    assert rates["naive_ratio"].rate.point < rates["delta_method_ratio"].rate.point


def test_the_negative_control_stays_quiet_while_a_real_effect_exists(ab):
    for check in negative_controls(ab):
        assert check.holds, f"{check.estimator} is reacting to a metric the treatment cannot move"


def test_a_saturating_metric_reports_less_than_the_stated_lift(ab):
    frame = attenuation(ab).set_index("metric")
    assert frame.loc["purchases per session", "share_of_stated_lift"] == pytest.approx(1.0, abs=0.1)
    assert frame.loc["users who converted", "share_of_stated_lift"] < 0.9
    assert abs(frame.loc["sessions per user", "share_of_stated_lift"]) < 0.05


def test_a_false_positive_rate_cannot_be_computed_from_a_study_with_an_effect(ab):
    """Measuring power and calling it a false positive rate is an easy and serious mistake."""
    with pytest.raises(ValueError):
        false_positive_rate(ab)


def test_every_estimator_declares_which_metric_it_measures():
    from pae.experiment.validate import ESTIMATORS
    for _, (metric, _) in ESTIMATORS.items():
        assert metric in METRIC_TRUTH

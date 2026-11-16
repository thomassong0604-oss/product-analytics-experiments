"""The generator is the measuring instrument. If its ground truth is wrong, every check built on it
is measuring nothing, so its invariants are tested harder than anything else here."""
import numpy as np
import pytest

from pae.generate import PURCHASE_SCALE, Spec, assignment, dataset, sessions, simulate_units, users


def test_the_stated_lift_is_exactly_recoverable():
    """The treatment multiplies the purchase probability, so the ratio of arm means must be 1+lift.

    Averaged over seeds, because one seed's deviation is sampling noise. This is the single property
    the whole validation study rests on.
    """
    for lift in (0.0, 0.20):
        observed = []
        for seed in range(12):
            spec = Spec(n_users=4_000, true_lift=lift, seed=20260101 + seed * 977)
            user_frame = users(spec)
            assign = assignment(spec, user_frame)
            frame = sessions(spec, user_frame, assign)
            rates = frame.groupby("variant").purchase.mean()
            observed.append(rates["treatment"] / rates["control"] - 1)
        mean = float(np.mean(observed))
        standard_error = float(np.std(observed, ddof=1) / len(observed) ** 0.5)
        assert abs(mean - lift) < 3 * standard_error, (
            f"stated {lift}, recovered {mean:.4f} +/- {standard_error:.4f}")


def test_no_probability_is_clipped():
    """A clipped probability would silently absorb the lift and make the truth unrecoverable."""
    spec = Spec(n_users=6_000, true_lift=0.5, base_purchase_rate=0.25)
    user_frame = users(spec)
    assert (user_frame.propensity * PURCHASE_SCALE * (1 + spec.true_lift)).max() <= 1.0


def test_the_treatment_moves_only_the_purchase_step():
    """An effect on an earlier step would mean the generator is modelling something else."""
    spec = Spec(n_users=30_000, true_lift=0.35)
    user_frame = users(spec)
    frame = sessions(spec, user_frame, assignment(spec, user_frame))
    grouped = frame.groupby("variant")
    for step in ("product_view", "add_to_cart", "checkout_start"):
        rates = grouped[step].mean()
        assert abs(rates["treatment"] / rates["control"] - 1) < 0.02, step
    purchase = grouped.purchase.mean()
    assert purchase["treatment"] / purchase["control"] - 1 > 0.2


def test_sessions_per_user_are_untouched_by_the_treatment():
    """The negative control in the validation study depends on this being exactly true."""
    spec = Spec(n_users=30_000, true_lift=0.35)
    units = simulate_units(spec)
    rates = units.groupby("variant").sessions.mean()
    assert abs(rates["treatment"] / rates["control"] - 1) < 0.03


def test_the_pre_period_is_balanced_because_it_precedes_the_treatment():
    units = simulate_units(Spec(n_users=20_000, true_lift=0.35))
    grouped = units.groupby("variant")
    for column in ("sessions_pre", "purchases_pre", "revenue_pre"):
        means = grouped[column].mean()
        assert abs(means["treatment"] / means["control"] - 1) < 0.06, column


def test_the_pre_period_covariate_actually_predicts_the_outcome():
    """CUPED can only reduce variance in proportion to this correlation. If a change to the
    generator breaks the persistence, CUPED would appear useless for the wrong reason."""
    units = simulate_units(Spec(n_users=20_000))
    assert np.corrcoef(units.sessions, units.sessions_pre)[0, 1] > 0.5
    assert np.corrcoef(units.revenue, units.revenue_pre)[0, 1] > 0.15


def test_latent_variables_never_reach_the_warehouse():
    """Propensity and activity are what the estimators are implicitly trying to average over.
    Shipping them would let a model condition on the answer."""
    tables = dataset(Spec(n_users=500))
    for name, frame in tables.items():
        assert "propensity" not in frame.columns, name
        assert "activity" not in frame.columns, name


def test_a_users_first_session_is_at_signup():
    """Otherwise week-zero cohort retention comes out below week one, which looks like a bug in the
    retention SQL rather than an artefact of the generator."""
    spec = Spec(n_users=2_000)
    user_frame = users(spec)
    frame = sessions(spec, user_frame, assignment(spec, user_frame))
    first = frame.sort_values("session_ts").groupby("user_id").session_ts.min()
    signup = user_frame.set_index("user_id").signup_date
    gap = (first - signup.loc[first.index].map(lambda d: __import__("pandas").Timestamp(d)))
    assert (gap.dt.total_seconds() < 86_400).mean() > 0.95


def test_generation_is_reproducible():
    a = dataset(Spec(n_users=800, seed=7))
    b = dataset(Spec(n_users=800, seed=7))
    for name in a:
        assert a[name].equals(b[name]), name


def test_a_different_seed_gives_different_data():
    a = dataset(Spec(n_users=800, seed=7))
    b = dataset(Spec(n_users=800, seed=8))
    assert not a["events"].equals(b["events"])


def test_the_event_stream_matches_the_session_table():
    """The event stream is derived from sessions, so a purchase event must exist for exactly the
    sessions flagged as purchases. A mismatch would make the warehouse and the validation study
    disagree about what happened."""
    spec = Spec(n_users=1_500)
    user_frame = users(spec)
    session_frame = sessions(spec, user_frame, assignment(spec, user_frame))
    tables = dataset(spec)
    purchases_in_events = set(
        tables["events"].query("event_name == 'purchase'").session_id)
    purchases_in_sessions = set(session_frame.query("purchase").session_id)
    assert purchases_in_sessions <= purchases_in_events


def test_ground_truth_names_the_metric_the_lift_applies_to():
    """The lift is applied to a session's purchase probability. Labelling it as a per-user rate
    would send the coverage study chasing the wrong number."""
    truth = dataset(Spec(n_users=400, true_lift=0.1))["ground_truth"].iloc[0]
    assert truth.metric == "purchase_rate_per_session"
    assert truth.true_relative_lift == pytest.approx(0.1)

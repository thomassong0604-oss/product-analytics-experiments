"""A synthetic event stream whose true treatment effect is known.

The point of generating data is not to avoid real data — the loaders in `pae.load` read real public
datasets. It is that a real dataset cannot tell you whether your estimator is correct, because the
true effect is unknown. Here it is an input, so every claim the analysis makes can be checked:

    does the point estimate land on the true lift?
    does the 95% interval contain it 95% of the time?
    with no effect at all, does the test reject 5% of the time?

Two properties of the generator matter for those checks and are deliberate:

  heterogeneity   each user carries a latent purchase propensity, so counts are overdispersed
                  relative to a binomial. Real users behave this way, and it is what makes a
                  naive variance estimate on a per-session metric too small.
  clustering      users return across several sessions. A metric defined per *session* therefore
                  has correlated observations inside each user, which is the single most common way
                  a real experiment readout understates its own uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

EXPERIMENT = "checkout_redesign"
FUNNEL = ["session_start", "product_view", "add_to_cart", "checkout_start", "purchase"]
# Funnel shape. The purchase step is scaled so that a user's probability stays well under 1 at every
# propensity the Beta can produce, which is what keeps the stated lift exactly recoverable.
VIEW_RATE = 0.93
CART_BASE, CART_SLOPE = 0.42, 0.9
CHECKOUT_BASE, CHECKOUT_SLOPE = 0.55, 0.7
PURCHASE_SCALE = 0.62
DEVICES = ["ios", "android", "desktop"]
DEVICE_P = [0.34, 0.38, 0.28]
CHANNELS = ["organic", "paid_search", "social", "email", "referral"]
CHANNEL_P = [0.38, 0.22, 0.18, 0.14, 0.08]
COUNTRIES = ["US", "GB", "CA", "DE", "AU"]
COUNTRY_P = [0.52, 0.14, 0.12, 0.13, 0.09]


@dataclass(frozen=True)
class Spec:
    """Everything the generator needs, and the ground truth the analysis will be checked against."""
    n_users: int = 20_000
    days: int = 56
    start: date = date(2026, 1, 5)
    seed: int = 20260101

    # --- ground truth -----------------------------------------------------------------------
    true_lift: float = 0.0
    """Relative lift the treatment applies to each user's purchase propensity. 0.0 makes this an
    A/A test, which is how the false-positive rate is measured."""
    treated_share: float = 0.5

    # --- shape of the population ------------------------------------------------------------
    base_purchase_rate: float = 0.18
    propensity_dispersion: float = 2.2
    """Beta concentration. Lower means users differ more from one another; at a very high value the
    population becomes homogeneous and a naive variance estimate would be right."""
    sessions_mean: float = 8.0
    activity_dispersion: float = 0.8
    """Gamma shape for a user's persistent activity level. Without this every user would draw their
    session count from the same rate, session counts would carry no user-level signal from one
    period to the next, and a pre-period covariate would be worthless — which is not how real
    product data behaves, and would quietly make CUPED look useless."""
    revenue_median: float = 42.0
    revenue_sigma: float = 0.85
    novelty_decay_days: float = 0.0
    """If positive, the treatment effect decays over this many days — the novelty effect that makes
    an early readout overstate a long-run lift."""
    pre_days: int = 28
    """Length of a pre-experiment window, used only as a CUPED covariate. The treatment does not
    exist yet in this window, which is exactly why adjusting on it cannot bias the estimate."""

    seasonality: tuple[float, ...] = field(
        default=(1.18, 1.05, 0.98, 0.96, 1.02, 0.92, 0.89))
    """Day-of-week multipliers on session volume, Monday first."""


def _rng(spec: Spec, stream: int) -> np.random.Generator:
    """One independent stream per table, so adding a column never shifts another table's draws."""
    return np.random.default_rng([spec.seed, stream])


def users(spec: Spec) -> pd.DataFrame:
    rng = _rng(spec, 1)
    n = spec.n_users
    # Beta propensities with the requested mean: users differ, and that difference persists across
    # their sessions. This is the source of the overdispersion the variance checks look for.
    a = spec.propensity_dispersion
    b = a * (1 - spec.base_purchase_rate) / spec.base_purchase_rate
    frame = pd.DataFrame({
        "user_id": [f"u{i:07d}" for i in range(n)],
        "propensity": rng.beta(a, b, n),
        # Mean one by construction, so `sessions_mean` keeps its meaning across dispersions.
        "activity": rng.gamma(spec.activity_dispersion, 1 / spec.activity_dispersion, n),
        "signup_date": [spec.start + timedelta(days=int(d))
                        for d in rng.integers(0, spec.days, n)],
        "device": rng.choice(DEVICES, n, p=DEVICE_P),
        "channel": rng.choice(CHANNELS, n, p=CHANNEL_P),
        "country": rng.choice(COUNTRIES, n, p=COUNTRY_P),
    })
    frame["cohort_week"] = pd.to_datetime(frame.signup_date).dt.to_period("W").dt.start_time.dt.date
    return frame


def assignment(spec: Spec, user_frame: pd.DataFrame) -> pd.DataFrame:
    """Randomise on the user, not the session.

    Assigning sessions would split one person across both variants and contaminate the comparison.
    Randomising the unit the metric is later aggregated over is the one design choice that cannot be
    repaired in analysis.
    """
    rng = _rng(spec, 2)
    variant = np.where(rng.random(len(user_frame)) < spec.treated_share, "treatment", "control")
    return pd.DataFrame({
        "user_id": user_frame.user_id,
        "experiment": EXPERIMENT,
        "variant": variant,
        "assigned_at": [datetime.combine(d, datetime.min.time()) for d in user_frame.signup_date],
    })


def sessions(spec: Spec, user_frame: pd.DataFrame, assign_frame: pd.DataFrame,
             pre_period: bool = False) -> pd.DataFrame:
    """One row per session, with each funnel step reached or not. The single source of truth.

    The event stream, the warehouse metrics and the fast simulator used by the replication study are
    all derived from this one function, so they cannot drift apart and quietly disagree about what
    the treatment did.

    The treatment multiplies the purchase step only, and nothing is clipped: if a probability were
    capped at 1 the high-propensity users would absorb the lift, the realised effect would be
    smaller than the stated one, and every check against the "true" value would be measuring the
    generator's saturation rather than the estimator.
    """
    rng = _rng(spec, 4 if pre_period else 3)
    variant = assign_frame.set_index("user_id").variant
    n = len(user_frame)

    # A user's activity level persists across both periods; only the Poisson draw around it is
    # fresh. That persistence is what a pre-period covariate picks up.
    n_sessions = 1 + rng.poisson((spec.sessions_mean - 1) * user_frame.activity.to_numpy(), n)
    user_ids = np.repeat(user_frame.user_id.to_numpy(), n_sessions)
    propensity = np.repeat(user_frame.propensity.to_numpy(), n_sessions)
    signup = np.repeat(pd.to_datetime(user_frame.signup_date).to_numpy(), n_sessions)
    is_treated = np.repeat((variant.loc[user_frame.user_id] == "treatment").to_numpy(), n_sessions)

    if pre_period:
        # Spread uniformly over the window before the experiment started. Every user is present,
        # whatever their signup date, so the covariate is defined for everyone.
        window_start = pd.Timestamp(spec.start - timedelta(days=spec.pre_days))
        timestamps = window_start + pd.to_timedelta(
            rng.random(len(user_ids)) * spec.pre_days * 86400, unit="s")
        signup = np.full(len(user_ids), window_start.to_datetime64())
        keep = timestamps < pd.Timestamp(spec.start)
    else:
        offset_days = rng.exponential(spec.days / 6.0, len(user_ids))
        # A user's first session happens at signup, because signing up *is* something you do during
        # a session. Without this, a user's first visit could land weeks after their signup date and
        # week-zero cohort retention would come out below week one — an artefact of the generator
        # that would look like a bug in the retention SQL.
        first_of_user = np.zeros(len(user_ids), dtype=bool)
        first_of_user[np.cumsum(n_sessions) - n_sessions] = True
        offset_days[first_of_user] = 0.0
        timestamps = signup + pd.to_timedelta(offset_days, unit="D") + pd.to_timedelta(
            rng.integers(0, 24 * 3600, len(user_ids)), unit="s")
        keep = timestamps < pd.Timestamp(spec.start + timedelta(days=spec.days))
    user_ids, propensity, is_treated = user_ids[keep], propensity[keep], is_treated[keep]
    timestamps, signup = timestamps[keep], signup[keep]
    total = len(user_ids)

    weekday = pd.DatetimeIndex(timestamps).weekday.to_numpy()
    season = np.asarray(spec.seasonality)[weekday]

    # No treatment exists in the pre-period, so no lift is applied there.
    lift = np.zeros(total) if pre_period else np.where(is_treated, spec.true_lift, 0.0)
    if spec.novelty_decay_days > 0 and not pre_period:
        age = (timestamps - signup) / np.timedelta64(1, "D")
        lift = lift * np.exp(-age / spec.novelty_decay_days)

    # Funnel step probabilities. Only the last responds to the treatment, so a readout that moved
    # every step would be a bug rather than a checkout redesign.
    p_view = np.clip(VIEW_RATE * season / 1.02, 0, 1)
    p_cart = np.clip(CART_BASE + CART_SLOPE * propensity, 0, 1)
    p_checkout = np.clip(CHECKOUT_BASE + CHECKOUT_SLOPE * propensity, 0, 1)
    p_purchase = propensity * PURCHASE_SCALE * (1 + lift)      # never clipped: see the docstring
    assert p_purchase.max() <= 1.0, "purchase probability must stay a probability"

    viewed = rng.random(total) < p_view
    carted = viewed & (rng.random(total) < p_cart)
    checked = carted & (rng.random(total) < p_checkout)
    purchased = checked & (rng.random(total) < p_purchase)
    revenue = np.where(purchased,
                       np.round(rng.lognormal(np.log(spec.revenue_median), spec.revenue_sigma,
                                              total), 2), 0.0)

    prefix = "p" if pre_period else "s"
    return pd.DataFrame({
        "session_id": [f"{prefix}{i:08d}" for i in range(total)],
        "user_id": user_ids,
        "variant": np.where(is_treated, "treatment", "control"),
        "session_ts": timestamps,
        "session_start": np.ones(total, bool),
        "product_view": viewed,
        "add_to_cart": carted,
        "checkout_start": checked,
        "purchase": purchased,
        "revenue": revenue,
    }).sort_values("session_ts").reset_index(drop=True)


def events(session_frame: pd.DataFrame) -> pd.DataFrame:
    """The session table unpivoted into the event stream a warehouse would actually receive.

    Only reached steps become rows, and each step is stamped a little later than the last, so the
    ordering a funnel query depends on is present in the data rather than assumed by the query.
    """
    rows = []
    for step, name in enumerate(FUNNEL):
        mask = session_frame[name].to_numpy()
        if not mask.any():
            continue
        part = session_frame.loc[mask, ["user_id", "session_id", "session_ts", "revenue"]].copy()
        part["event_name"] = name
        part["funnel_step"] = step
        part["event_ts"] = part.session_ts + pd.to_timedelta(step * 37, unit="s")
        part["revenue"] = part.revenue if name == "purchase" else 0.0
        rows.append(part.drop(columns=["session_ts"]))
    frame = pd.concat(rows, ignore_index=True).sort_values(["event_ts", "session_id", "funnel_step"])
    frame.insert(0, "event_id", [f"e{i:09d}" for i in range(len(frame))])
    return frame[["event_id", "user_id", "session_id", "event_name", "funnel_step", "event_ts",
                  "revenue"]].reset_index(drop=True)


def simulate_units(spec: Spec | None = None) -> pd.DataFrame:
    """One row per user: the unit of analysis, aggregated from the same sessions the warehouse sees.

    This is the fast path the replication study calls thousands of times. It shares `sessions()`
    with the full pipeline, so the distribution it samples is the distribution the warehouse holds —
    not a simplified stand-in that might behave better than the real thing.
    """
    spec = spec or Spec()
    user_frame = users(spec)
    assign_frame = assignment(spec, user_frame)
    session_frame = sessions(spec, user_frame, assign_frame)
    pre_frame = sessions(spec, user_frame, assign_frame, pre_period=True)

    def roll(frame, suffix=""):
        agg = frame.groupby("user_id", observed=True).agg(
            sessions=("session_id", "size"),
            purchases=("purchase", "sum"),
            revenue=("revenue", "sum"),
            carts=("add_to_cart", "sum"),
        )
        return agg.add_suffix(suffix) if suffix else agg

    out = (assign_frame.set_index("user_id")
           .join(roll(session_frame))
           .join(roll(pre_frame, "_pre"))
           .fillna({"sessions": 0, "purchases": 0, "revenue": 0.0, "carts": 0,
                    "sessions_pre": 0, "purchases_pre": 0, "revenue_pre": 0.0, "carts_pre": 0}))
    out["converted"] = out.purchases > 0
    return out.reset_index()


def dataset(spec: Spec | None = None) -> dict[str, pd.DataFrame]:
    """The tables the warehouse ingests, plus the ground truth the checks compare against."""
    spec = spec or Spec()
    user_frame = users(spec)
    assign_frame = assignment(spec, user_frame)
    # Both periods land in the warehouse. The pre-period is what gives CUPED a covariate downstream;
    # leaving it out of here while `simulate_units` generated it would mean the warehouse and the
    # validation study were analysing different data, and CUPED would silently do nothing in one of
    # them. They are concatenated before `events` so event ids stay unique across the whole stream.
    session_frame = pd.concat([
        sessions(spec, user_frame, assign_frame, pre_period=True),
        sessions(spec, user_frame, assign_frame),
    ], ignore_index=True).sort_values("session_ts").reset_index(drop=True)
    event_frame = events(session_frame)
    truth = pd.DataFrame([{
        "experiment": EXPERIMENT,
        # The effect is applied to the purchase step of a session, so this — and not the per-user
        # conversion rate — is the quantity the stated lift refers to. Per-user conversion is a
        # saturating function of it ("purchased at least once"), so its lift is smaller, and
        # checking an estimator of it against this number would fail for a reason that has nothing
        # to do with the estimator.
        "metric": "purchase_rate_per_session",
        "true_relative_lift": spec.true_lift,
        "base_purchase_rate": spec.base_purchase_rate,
        "sessions_mean": spec.sessions_mean,
        "treated_share": spec.treated_share,
        "novelty_decay_days": spec.novelty_decay_days,
        "seed": spec.seed,
        "n_users": spec.n_users,
    }])
    # `propensity` and `activity` are the latent variables the analysis is not allowed to see. They
    # stay out of the warehouse: leaving them in would let a model condition on the very thing being
    # estimated, and every check downstream would be measuring nothing.
    return {"users": user_frame.drop(columns=["propensity", "activity"]),
            "assignment": assign_frame, "events": event_frame, "ground_truth": truth}

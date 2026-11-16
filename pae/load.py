"""Loading real public datasets into the same shape the synthetic generator produces.

Why both. A real dataset has the mess that matters — returns, duplicate invoices, missing customer
ids, a long tail of tiny customers — and exercising the warehouse models against it is the only way
to know they survive contact with reality. What a real dataset cannot give you is a known treatment
effect, so it cannot tell you whether an estimator is correct.

So the two sources do different jobs, and this module is explicit about the limit:

    synthetic   known true effect -> can check bias, coverage and the false positive rate
    real        unknown effect    -> can check that the pipeline runs, and can run an A/A test

The A/A test on real data is worth more than it sounds. Splitting real users at random creates two
arms that genuinely differ in nothing, so an estimator that rejects the null more than 5% of the
time on *this* data is mis-specified for this data — heavier tails, more extreme clustering, or a
grain that does not match. That is a calibration check no amount of synthetic data can substitute
for, because the only thing it holds constant is the thing being tested.

Any `variant` column this module produces is therefore a random split and never a real experiment.
It is labelled `aa_random_split` so a readout built on it cannot be mistaken for a shipped test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

AA_EXPERIMENT = "aa_random_split"


def _assign(user_ids: pd.Series, seed: int, treated_share: float = 0.5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "user_id": user_ids.to_numpy(),
        "experiment": AA_EXPERIMENT,
        "variant": np.where(rng.random(len(user_ids)) < treated_share, "treatment", "control"),
        "assigned_at": pd.Timestamp("1970-01-01"),
    })


def online_retail_ii(path: str | Path, seed: int = 20260101) -> dict[str, pd.DataFrame]:
    """UCI Online Retail II — two years of real transactions from a UK online retailer.

    Download it yourself (it is not redistributed here):
        https://archive.ics.uci.edu/dataset/502/online+retail+ii

    The mapping to an event stream is a judgement call and is stated rather than hidden:

      user      `Customer ID`. Rows without one are guest checkouts that cannot be tied to a
                person, so they are dropped — they would otherwise all collapse into one enormous
                "user" and dominate every per-user metric.
      session   the invoice. A basket is the closest thing this dataset has to a visit; it is not a
                browsing session, so the funnel steps above purchase do not exist here and the
                funnel mart will only show the purchase step.
      purchase  one event per invoice, with revenue summed over its lines.
      returns   negative quantities are credit notes. They are kept, and netted into revenue, so a
                refunded order does not count as revenue the business never earned.

    The experiment column is a random split, not a real test. See the module docstring.
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        frames = pd.read_excel(path, sheet_name=None)
        raw = pd.concat(frames.values(), ignore_index=True)
    else:
        raw = pd.read_csv(path, encoding="latin-1")

    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    customer = "customer_id" if "customer_id" in raw.columns else "customerid"
    invoice = "invoice" if "invoice" in raw.columns else "invoiceno"
    price = "price" if "price" in raw.columns else "unitprice"

    raw = raw.dropna(subset=[customer])
    raw["user_id"] = "u" + raw[customer].astype("int64").astype(str)
    raw["session_id"] = "i" + raw[invoice].astype(str)
    raw["event_ts"] = pd.to_datetime(raw["invoicedate"])
    raw["line_revenue"] = raw["quantity"] * raw[price]

    baskets = (raw.groupby(["user_id", "session_id"], observed=True)
               .agg(event_ts=("event_ts", "min"), revenue=("line_revenue", "sum"),
                    country=("country", "first"))
               .reset_index())

    events = pd.DataFrame({
        "event_id": [f"e{i:09d}" for i in range(len(baskets))],
        "user_id": baskets.user_id,
        "session_id": baskets.session_id,
        "event_name": "purchase",
        "funnel_step": 4,
        "event_ts": baskets.event_ts,
        "revenue": baskets.revenue.round(2),
    })
    # A session_start for every basket, so session counts and the funnel's denominator exist.
    starts = events.copy()
    starts["event_id"] = [f"s{i:09d}" for i in range(len(starts))]
    starts["event_name"] = "session_start"
    starts["funnel_step"] = 0
    starts["revenue"] = 0.0
    starts["event_ts"] = starts.event_ts - pd.Timedelta(seconds=1)
    events = pd.concat([starts, events], ignore_index=True).sort_values("event_ts")

    first_seen = baskets.groupby("user_id", observed=True).agg(
        signup_date=("event_ts", "min"), country=("country", "first")).reset_index()
    users = pd.DataFrame({
        "user_id": first_seen.user_id,
        "signup_date": first_seen.signup_date.dt.date,
        "cohort_week": first_seen.signup_date.dt.to_period("W").dt.start_time.dt.date,
        "device": "unknown",      # the dataset has none; stated rather than invented
        "channel": "unknown",
        "country": first_seen.country,
    })

    assignment = _assign(users.user_id, seed)
    # The assignment timestamp sets the pre/experiment boundary in int_user_period. Put it at the
    # median basket date so an A/A split has a real pre-period to draw a CUPED covariate from.
    assignment["assigned_at"] = baskets.event_ts.median()

    truth = pd.DataFrame([{
        "experiment": AA_EXPERIMENT,
        "metric": "purchase_rate_per_session",
        "true_relative_lift": 0.0,       # a random split of real users has no effect, by design
        "source": "UCI Online Retail II",
        "note": "random split of real users: an A/A test, not a shipped experiment",
        "seed": seed,
        "n_users": len(users),
    }])
    return {"users": users, "assignment": assignment, "events": events, "ground_truth": truth}

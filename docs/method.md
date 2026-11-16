# Method

The judgement calls behind the numbers, and what each one costs. Anything a reader might reasonably
object to should be here, named, with the objection stated.

## 1. Why there is a generator at all

A real dataset cannot tell you whether your estimator is correct, because its true effect is
unknown. You can run an analysis on real data, get an interval, and have no way to find out whether
that interval would contain the truth 95% of the time or 87% of the time. The difference matters:
87% coverage means roughly one readout in eight is confidently wrong.

So the project has both. The warehouse models run against a real public dataset
(`pae.load.online_retail_ii`) because that is where the mess lives — returns, duplicate invoices,
missing customer ids. The estimators are checked against generated data because that is the only
place the answer is known. Neither substitutes for the other, and `pae/load.py` says so where
someone might assume otherwise.

The generator's parameters are not tuned to flatter the analysis. They are set so that the data has
the two properties that make product metrics hard, both documented in `pae/generate.py`:

- **heterogeneity** — users carry a persistent purchase propensity, so counts are overdispersed
  relative to a binomial.
- **clustering** — users return across several sessions, so a per-session metric has correlated
  observations inside each user.

Remove either and the naive estimator stops being wrong, and the central finding of the project
would evaporate for the wrong reason.

## 2. What the stated lift means, exactly

The generator multiplies the probability that a session ends in a purchase by `1 + true_lift`, and
touches nothing else. Three consequences, each verified by a test:

- **Purchases per session** moves by exactly the stated lift. No probability is clipped, so the
  ratio of arm means is exactly `1 + lift`. If a probability were capped at 1, the high-propensity
  users would absorb the lift, the realised effect would be smaller than the stated one, and every
  check against "the truth" would be measuring the generator's saturation instead of the estimator.
  This was a real bug before `PURCHASE_SCALE` was set low enough to make it impossible.
- **Revenue per user** also moves by exactly the stated lift, because revenue per user is
  `E[sessions] × P(purchase | session) × E[order value]` and only the middle term changes.
- **Sessions per user** does not move at all. It is the negative control.

**Per-user conversion is deliberately excluded from the coverage check.** "Purchased at least once"
is `1 - (1 - q)^n`, which saturates: a lift on `q` arrives attenuated by an amount that depends on
the session distribution, so the stated lift is not its true lift. Judging an estimator of it
against the stated number would convict a correct estimator. Instead `attenuation()` reports how
much the metric shrinks the effect — roughly 70% of the stated lift at these parameters — which is
a result worth having on its own: **the choice of metric alone changes the headline number by a
third**, before anyone argues about statistics.

## 3. Why the naive estimator is kept

`naive_ratio` is wrong and ships anyway, because the project's main claim is that it is wrong by a
specific, measurable amount. It computes exactly what `SUM(purchases) / SUM(sessions)` with a
binomial standard error gives you, which is what a SQL-first readout produces by default.

Running it beside the clustered estimator turns an argument into a number: the two agree on the
point estimate to the last decimal and differ only in the standard error, so the gap between their
false positive rates is attributable to the variance calculation and to nothing else.

## 4. The honesty ratio

The clearest single diagnostic in the validation report, and the one that needs no distributional
assumption:

```
honesty ratio = mean standard error the estimator reported
                ─────────────────────────────────────────────
                actual spread of its estimates across replications
```

An honest estimator lands at 1.0. Below 1.0 it is claiming more precision than it has. This works
because the replication study generates fresh data each time, so the spread of estimates *is* the
sampling distribution — the thing a standard error is supposed to be estimating. The naive
session-level estimator comes in near 0.79; everything else is within a few percent of 1.0.

## 5. Sequential boundaries are computed, not quoted

Checking a running experiment daily and stopping when `p < 0.05` does not give a 5% false positive
rate; with fourteen looks it is about 22%. The fix is to spend the error budget deliberately.

The Pocock and O'Brien-Fleming boundaries here are **calibrated numerically** rather than taken
from a table or from a closed form. The *shape* comes from the spending family; the *scale* is
found by bisection against an exact crossing probability computed with the Armitage-McPherson-Rowe
recursion — carrying the density of the partial sums forward through each look, truncated to the
continuation region.

An earlier version multiplied per-look error rates as if the looks were independent. They are
strongly correlated, and the resulting boundaries missed their nominal rate in both directions:
Pocock too conservative, O'Brien-Fleming above 5%. The recursion now reproduces the published
boundaries (Pocock 2.178 / 2.289 / 2.413 for two, three and five looks; O'Brien-Fleming final
values 1.977 / 2.004 / 2.040) to within 0.003 without having them hardcoded — which is what makes
the integration itself trustworthy — and the tests assert that.

## 6. CUPED, and when it is not worth doing

CUPED subtracts off the part of the outcome a pre-experiment covariate already explained:
`Y' = Y − θ(X − E[X])`. Because `X` precedes assignment it cannot have been affected by the
treatment, so the adjustment is unbiased. `θ` is fitted on the pooled data, not per arm — fitting
it separately lets the treatment influence its own adjustment and reintroduces the bias.

The benefit is about `ρ²`, so it is worth reporting honestly rather than universally:

| covariate correlation | standard error reduction |
|---|---|
| sessions per user, ρ ≈ 0.83 | large |
| revenue per user, ρ ≈ 0.33 | a few percent |
| a sparse metric, ρ ≈ 0.05 | nothing |

An early version of the generator gave every user the same session rate, so nothing persisted
across periods, every correlation was near zero and CUPED appeared useless. That was a flaw in the
data, not in the method — which is itself the point: CUPED is only as good as the persistence in
the metric, and a team seeing no benefit should check their covariate before blaming the technique.

## 7. The warehouse

DuckDB and dbt, because the whole project has to run on a laptop and in CI with no credentials. The
SQL is ordinary enough that the models move to BigQuery or Snowflake with a dialect change and a
different `profiles.yml`.

Three decisions worth defending:

- **De-duplication lives in staging**, not in each downstream model, so a double-counted purchase
  cannot reach a metric and the rule is testable in one place.
- **`fct_user_metrics` is one row per randomised user.** Aggregating to the randomised unit in the
  warehouse is what makes the honest standard error the easy one to compute downstream.
- **The pre/experiment boundary is read from the data** (`min(assigned_at)`), never hardcoded, so
  reseeding with different dates cannot silently move pre-period sessions inside the experiment.

Every one of those is enforced by a dbt test, including a cross-model test asserting that the daily
curve and the user-level fact table describe the same experiment. They drifted apart once: the
daily model was including pre-experiment sessions, which diluted the effect and reversed its sign
while every number still looked plausible in isolation. That is the failure mode a dashboard is
most likely to have and least likely to reveal.

## 8. What this project does not do

No multi-armed tests, no multiple-comparison correction across metrics, no heterogeneous treatment
effects or CUPED with several covariates, no switchback or cluster-randomised designs, no Bayesian
alternative. The ratio-metric variance uses the first-order delta method, which the cluster
bootstrap agrees with at these sample sizes but which would need checking at much smaller ones.

The "experiment" on real data is a random split — an A/A test — and never a shipped experiment.
`pae/load.py` labels it `aa_random_split` so a readout built on it cannot be mistaken for one.

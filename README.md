# product-analytics-experiments

**A product-analytics warehouse, and an experiment-analysis library that checks itself against a
known answer.**

Most A/B write-ups end with a p-value and no way to tell whether the analysis behind it was right.
Run the same estimator on 600 experiments where the true effect is known, though, and you can ask
the only question that matters: *does its 95% interval actually contain the truth 95% of the time?*

For the standard SQL readout — `SUM(purchases) / SUM(sessions)` with a binomial standard error —
the answer is **no**. It contains the truth 88% of the time and calls a null result significant
**12.2%** of the time instead of 5%. This repository measures that, explains it, fixes it, and
builds the warehouse that produces the numbers honestly.

[![tests](https://github.com/thomassong0604-oss/product-analytics-experiments/actions/workflows/ci.yml/badge.svg)](https://github.com/thomassong0604-oss/product-analytics-experiments/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![stack](https://img.shields.io/badge/stack-dbt%20%C2%B7%20DuckDB%20%C2%B7%20SQL-informational)
![license](https://img.shields.io/badge/license-MIT-green)

**[→ Live dashboard](https://thomassong0604-oss.github.io/product-analytics-experiments/)**  ·
**[→ Validation report](results/validation.md)**  ·  **[→ Method](docs/method.md)**

---

## The result

600 replications per study, 6,000 users each. Every estimator sees the same data; they differ only
in how they compute a standard error.

| estimator | false positives (want 0.05) | interval coverage (want 0.95) | honesty ratio |
|---|---|---|---|
| `SUM(y)/SUM(n)`, sessions as independent | **0.122** [0.098, 0.150] | **0.882** [0.853, 0.905] | **0.79** |
| delta method, clustered on user | 0.042 [0.028, 0.061] | 0.947 [0.926, 0.962] | 0.99 |
| two-proportion, per user | 0.055 [0.039, 0.076] | — | 1.03 |
| Welch, revenue per user | 0.048 [0.034, 0.069] | 0.940 [0.918, 0.956] | 1.00 |
| CUPED, revenue per user | 0.043 [0.030, 0.063] | 0.940 [0.918, 0.956] | 0.99 |

The **honesty ratio** is the clearest number here, and it needs no distributional assumption: the
standard error an estimator reported, divided by how much its estimate actually moved across
replications. An honest estimator lands at 1.0. The session-level one lands at 0.79 — it claims
21% more precision than it has, because a user's sessions are correlated and it treats them as
independent observations.

Both rows share a point estimate to the last decimal. **The entire difference is the variance
calculation.**

### Two more findings from the same study

**Peeking costs more than people expect.** Checking a running test every day for two weeks and
stopping at the first `p < 0.05`:

| stopping rule | false positive rate |
|---|---|
| stop at the first look where p < 0.05 | **0.222** |
| one analysis at a fixed horizon | 0.050 |
| Pocock boundary | 0.051 |
| O'Brien-Fleming boundary | 0.049 |

The boundaries are computed here by exact numerical integration, not copied from a table — and they
reproduce the published Pocock and O'Brien-Fleming values to three decimals, which is what makes
the integration trustworthy.

**The choice of metric moves the headline by a third.** The same +15% lift on the purchase step,
read on four metrics:

| metric | reported lift | share of the real effect |
|---|---|---|
| purchases per session | +15.08% | 1.01 |
| revenue per user | +15.20% | 1.01 |
| users who converted | **+10.17%** | **0.68** |
| sessions per user *(negative control)* | −0.01% | 0.00 |

"Purchased at least once" saturates — it cannot rise much among people who already buy — so it
reports a third less than the effect that actually happened. No statistics were argued over to lose
that third; someone just picked a metric.

---

## What's in it

```
pae/generate.py        event stream with a known true effect, and the two properties that make
                       product metrics hard: heterogeneous users, clustered sessions
pae/load.py            the same shape from a real public dataset (UCI Online Retail II)
dbt/                   staging -> intermediate -> marts, with 26 data tests
sql/                   standalone analysis: SRM check, RFM segments, funnel by segment, churn gaps
pae/experiment/        estimators, power, sequential boundaries, and the replication study
dashboard/             a self-contained HTML dashboard built from the warehouse
```

**The warehouse.** dbt over DuckDB, so it runs on a laptop and in CI with no credentials. Staging
de-duplicates the event stream in one place so a replayed purchase cannot reach a metric;
`int_sessions` assembles funnel flags and session ordering with window functions;
`fct_user_metrics` lands at one row per randomised user — the grain that makes the honest standard
error the easy one to compute downstream.

**26 dbt tests**, including a cross-model test that the daily curve and the user-level fact table
describe the same experiment. They drifted apart once: the daily model was including pre-experiment
sessions, which diluted the effect and *reversed its sign* while every number still looked
plausible in isolation. That is the failure a dashboard is least likely to reveal.

**The analysis.** Estimators that differ by the variance that belongs to them, a paired comparison
of each against a known truth, power and minimum-detectable-effect reported *beside* every result
so a null is never mistaken for a flat feature, and sequential boundaries for teams who will look
at the dashboard every morning regardless of what the methodology says.

---

## Running it

```bash
pip install -r requirements.txt
./scripts/run_all.sh --quick      # ~1 minute, no network, no credentials
./scripts/run_all.sh              # full size: 120k users, 5M events, 600 replications
```

That seeds the warehouse, runs dbt with its tests, produces the readout, runs the validation study
and rebuilds the dashboard. To run the models against real transactions instead:

```bash
# https://archive.ics.uci.edu/dataset/502/online+retail+ii  (not redistributed here)
python scripts/seed.py --source online-retail-ii --path ~/Downloads/online_retail_II.xlsx
./scripts/run_dbt.sh
```

```bash
python -m pytest -q      # 76 tests, ~40 seconds, includes a real dbt build
```

---

## What this work demonstrates

| skill | where to look |
|---|---|
| Analytics engineering with dbt | [staging → intermediate → marts](dbt/models) · [26 data tests](dbt/tests) |
| SQL beyond `GROUP BY` | [window functions and session ordering](dbt/models/intermediate/int_sessions.sql) · [cohort retention](dbt/models/marts/mart_retention.sql) · [RFM quintiles](sql/rfm_segments.sql) |
| Experiment design | [power and MDE](pae/experiment/power.py) · [sample ratio mismatch](sql/experiment_srm_check.sql) · [randomising on the right unit](pae/generate.py) |
| Knowing which standard error is correct | [delta method on a ratio metric](pae/experiment/estimate.py) · [cluster bootstrap as a cross-check](pae/experiment/estimate.py) |
| Variance reduction | [CUPED with a pooled θ](pae/experiment/estimate.py), and [when it is not worth doing](docs/method.md) |
| Sequential testing | [boundaries by exact numerical integration](pae/experiment/sequential.py) |
| Validating your own methods | [the replication study](pae/experiment/validate.py) · [its report](results/validation.md) |
| Data visualisation | [hand-written SVG, CVD-validated palette, light and dark](dashboard/build_dashboard.py) |
| Testing what quietly corrupts results | [76 tests](tests) · [each one is a bug that happened](tests/test_generate.py) |
| Reproducibility | [one command](scripts/run_all.sh) · [offline CI of the whole pipeline](.github/workflows/ci.yml) |

---

## Scope and honesty

- **The headline numbers come from generated data, and that is the point.** A real dataset has no
  known true effect, so it cannot tell you whether an estimator is calibrated. The generator's
  parameters are set to make the data realistically hard — heterogeneous users, clustered sessions —
  not to flatter the analysis; remove either property and the naive estimator stops being wrong.
- **The warehouse models also run on real data** (UCI Online Retail II, downloaded by you, not
  redistributed here). Splitting those real users at random is an A/A test, which is a calibration
  check generated data cannot substitute for. It is labelled `aa_random_split` so it can never be
  mistaken for a shipped experiment.
- **Limits are listed rather than left out**: no multi-armed tests, no multiple-comparison
  correction across metrics, no heterogeneous treatment effects, no switchback designs. The
  ratio-metric variance uses the first-order delta method, cross-checked against a cluster
  bootstrap. See [docs/method.md](docs/method.md), which also records the bugs this project had and
  what each one would have cost.

MIT licensed.

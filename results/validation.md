# Validation report

600 replications per study, 6,000 users each. Every rate below is itself an estimate, so it carries a Wilson interval; a verdict is given only when the nominal value falls outside it.


### False positive rate, A/A (nominal 0.05)

| estimator | metric | variance treated as | rejection rate | verdict |
|---|---|---|---|---|
| `naive_ratio` | purchases per session | sessions treated as independent | 0.122 [0.098, 0.150] (n=600) | **miscalibrated** |
| `two_proportion` | users who converted | one row per randomised unit | 0.055 [0.039, 0.076] (n=600) | as promised |
| `welch_mean` | revenue per user | unequal variances | 0.048 [0.034, 0.069] (n=600) | as promised |
| `cuped_revenue` | revenue per user | CUPED on pre-period revenue | 0.043 [0.030, 0.063] (n=600) | as promised |
| `delta_method_ratio` | purchases per session | clustered on user | 0.042 [0.028, 0.061] (n=600) | as promised |
| `cuped_sessions` | sessions per user | CUPED on pre-period sessions | 0.042 [0.028, 0.061] (n=600) | as promised |

### Interval coverage, A/B (nominal 0.95)

| estimator | metric | coverage | verdict |
|---|---|---|---|
| `naive_ratio` | purchases per session | 0.882 [0.853, 0.905] (n=600) | **miscalibrated** |
| `welch_mean` | revenue per user | 0.940 [0.918, 0.956] (n=600) | as promised |
| `cuped_revenue` | revenue per user | 0.940 [0.918, 0.956] (n=600) | as promised |
| `delta_method_ratio` | purchases per session | 0.947 [0.926, 0.962] (n=600) | as promised |
| `cuped_sessions` | sessions per user | 0.958 [0.939, 0.972] (n=600) | as promised |

### Does each estimator's claimed precision match how much it actually moves?

An honesty ratio below 1 means the estimator reports a standard error smaller than the spread of its own estimates across replications — it is overstating its own precision.

| estimator | claimed SE | actual spread | honesty ratio | unbiased |
|---|---|---|---|---|
| `naive_ratio` | 0.0447 | 0.0565 | **0.79** | yes |
| `delta_method_ratio` | 0.0559 | 0.0565 | **0.99** | yes |
| `cuped_revenue` | 0.0749 | 0.0755 | **0.99** | yes |
| `welch_mean` | 0.0784 | 0.0786 | **1.00** | yes |
| `cuped_sessions` | 0.0153 | 0.0151 | **1.01** | yes |
| `two_proportion` | 0.0493 | 0.0480 | **1.03** | yes |

### Negative control: a metric the treatment cannot move

While a real effect exists elsewhere, an estimator run on sessions per user should still find nothing. If it does find something, the effect is leaking into a metric it has no mechanism to touch.

| estimator | metric | rejection rate | verdict |
|---|---|---|---|
| `cuped_sessions` | sessions per user | 0.042 [0.028, 0.061] (n=600) | as promised |

### How much the choice of metric changes the headline

The generator applied a +15% lift to the probability that a session ends in a purchase. Read on a different metric, the same effect reports a different number — and on a saturating one it reports a smaller one.

| metric | observed lift | share of the stated lift |
|---|---|---|
| purchases per session | +15.08% [+14.72%, +15.44%] | 1.01 |
| users who converted | +10.17% [+9.76%, +10.57%] | 0.68 |
| revenue per user | +15.20% [+14.71%, +15.70%] | 1.01 |
| sessions per user | -0.01% [-0.13%, +0.11%] | -0.00 |

### What peeking costs, and what fixes it (14 looks)

Stopping at the first look where p < 0.05 does not give a 5% false positive rate. The boundaries are calibrated by exact numerical integration, so they hold their nominal rate rather than approximately holding it.

| stopping rule | false positive rate |
|---|---|
| stop at the first look where p < 0.05 | **0.222** |
| one analysis at a fixed horizon | 0.050 |
| Pocock boundary | 0.051 |
| O'Brien-Fleming boundary | 0.049 |

### Power to detect the stated lift

| estimator | metric | power |
|---|---|---|
| `naive_ratio` | purchases per session | 0.830 [0.798, 0.858] (n=600) |
| `delta_method_ratio` | purchases per session | 0.713 [0.676, 0.748] (n=600) |
| `two_proportion` | users who converted | 0.533 [0.493, 0.573] (n=600) |
| `cuped_revenue` | revenue per user | 0.490 [0.450, 0.530] (n=600) |
| `welch_mean` | revenue per user | 0.452 [0.412, 0.492] (n=600) |
| `cuped_sessions` | sessions per user | 0.042 [0.028, 0.061] (n=600) |

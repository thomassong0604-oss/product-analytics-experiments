"""What repeated looks do to a p-value, and how to look anyway.

Checking a running experiment every morning and stopping when p < 0.05 does not give a 5% false
positive rate. It gives whatever rate that stopping rule produces, which for daily looks over two
weeks is in the region of 20-30%. The honest options are to fix the horizon in advance, or to spend
the error budget across looks deliberately. Both are here; `pae.experiment.validate` measures the
first and confirms the second.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy import stats

ALPHA_SPENDING = ("pocock", "obrien-fleming")


def _crossing_probability(z_boundaries: list[float] | np.ndarray,
                          grid_points: int = 2001) -> float:
    """Exact probability that a null statistic crosses these boundaries at some look.

    Computed by the Armitage-McPherson-Rowe recursion rather than by simulation. With K equally
    spaced looks the partial sums S_k form a random walk with unit-variance increments, and the
    boundary |Z_k| > c_k is |S_k| > c_k·√k. The density of S_k restricted to the continuation region
    is carried forward one look at a time:

        f₁(s) = φ(s)                                      on |s| < b₁
        f_k(s) = ∫ f_{k-1}(u)·φ(s − u) du                 on |s| < b_k

    and the crossing probability is one minus the mass that survives to the final look. The looks
    are strongly correlated, so a closed form that assumes otherwise is not an alternative: an
    earlier version of this file multiplied per-look error rates as if independent and produced
    boundaries that missed their nominal rate in both directions.
    """
    z = np.asarray(z_boundaries, dtype=float)
    looks = z.size
    information = np.arange(1, looks + 1, dtype=float)
    boundary = z * np.sqrt(information)
    extent = float(boundary.max())
    grid = np.linspace(-extent, extent, grid_points)
    step = grid[1] - grid[0]

    density = stats.norm.pdf(grid)
    density[np.abs(grid) > boundary[0]] = 0.0
    if looks > 1:
        # Increments are identically distributed, so one kernel serves every transition.
        kernel = stats.norm.pdf(grid[:, None] - grid[None, :])
        for k in range(1, looks):
            density = kernel @ density * step
            density[np.abs(grid) > boundary[k]] = 0.0
    return float(1.0 - density.sum() * step)


def _calibrate(shape: np.ndarray, alpha: float, grid_points: int = 1201) -> float:
    """Scale a boundary shape until its exact crossing probability equals alpha.

    The grid is coarser here than the default used for verification. At 1201 points the published
    Pocock and O'Brien-Fleming boundaries are reproduced to within 0.002 in z, which is far finer
    than the choice between spending families matters — and it keeps the calibration fast enough to
    run inside a test suite.
    """
    lo, hi = 0.5, 8.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if _crossing_probability(shape * mid, grid_points) > alpha:
            lo = mid                        # boundary too low, too many crossings
        else:
            hi = mid
    return (lo + hi) / 2


@lru_cache(maxsize=256)
def sequential_boundaries(looks: int, alpha: float = 0.05,
                          spending: str = "obrien-fleming") -> tuple[float, ...]:
    """Critical z values for each interim look, so the overall error rate is exactly `alpha`.

    The *shape* comes from the spending family and the *scale* is calibrated numerically, so the
    boundaries hold their nominal rate instead of approximately holding it.

    Pocock keeps one constant z across looks: early stopping is easier, at the cost of a stricter
    final look. O'Brien-Fleming starts almost unreachable and relaxes to nearly the unadjusted
    threshold at the end, so the final analysis is barely penalised — usually the better trade for a
    product experiment, where stopping on day two is rarely the goal.
    """
    if looks < 1:
        raise ValueError("need at least one look")
    if spending not in ALPHA_SPENDING:
        raise ValueError(f"unknown spending function {spending!r}, expected one of {ALPHA_SPENDING}")
    information = np.arange(1, looks + 1, dtype=float) / looks
    shape = (np.ones(looks) if spending == "pocock" else 1.0 / np.sqrt(information))
    return tuple(float(round(z, 4)) for z in shape * _calibrate(shape, alpha))


def peeking_alpha(looks: int, alpha: float = 0.05, iterations: int = 20_000,
                  seed: int = 20260101) -> dict:
    """The false positive rate of stopping at the first look where p < alpha.

    Simulated rather than quoted: a Brownian path is accumulated in `looks` equal increments under
    a true null, and the run counts as a rejection if the standardised statistic ever crosses the
    unadjusted boundary. The same paths are then tested against the corrected boundaries, so the
    comparison is like for like rather than two separate simulations.
    """
    from .power import _z
    rng = np.random.default_rng(seed)
    z_fixed = _z(1 - alpha / 2)
    pocock = list(sequential_boundaries(looks, alpha, "pocock"))
    obf = list(sequential_boundaries(looks, alpha, "obrien-fleming"))

    increments = rng.normal(size=(iterations, looks))
    cumulative = np.cumsum(increments, axis=1)
    information = np.sqrt(np.arange(1, looks + 1))
    statistic = np.abs(cumulative / information)         # standardised at each look

    return {
        "looks": looks, "alpha": alpha, "iterations": iterations,
        "naive_peeking": float(np.mean(np.any(statistic > z_fixed, axis=1))),
        "fixed_horizon": float(np.mean(statistic[:, -1] > z_fixed)),
        "pocock": float(np.mean(np.any(statistic > np.array(pocock), axis=1))),
        "obrien_fleming": float(np.mean(np.any(statistic > np.array(obf), axis=1))),
    }

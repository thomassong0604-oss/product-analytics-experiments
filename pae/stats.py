"""Intervals for the rates this project reports about itself.

A false positive rate measured over 500 replications is itself an estimate, and quoting it bare
invites exactly the overconfidence the replication study exists to expose. So every rate here —
including "the rejection rate was 0.052" — carries an interval, and a verdict is only given when
the nominal value falls outside it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

Z = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


@dataclass(frozen=True)
class Rate:
    successes: int
    n: int
    low: float
    high: float

    @property
    def point(self) -> float:
        return self.successes / self.n if self.n else float("nan")

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def __str__(self) -> str:
        if not self.n:
            return "n/a"
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}] (n={self.n})"


def wilson(successes: int, n: int, conf: float = 0.95) -> Rate:
    """Wilson score interval.

    Chosen over the normal approximation because the rates here sit near 0.05 and 0.95, where the
    normal interval misbehaves: it can run outside [0, 1], and at a rate of exactly 0 or 1 it
    collapses to zero width and claims certainty from a few hundred replications.
    """
    if n <= 0:
        return Rate(successes, 0, float("nan"), float("nan"))
    z = Z.get(conf, 1.96)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Rate(successes, n, max(0.0, centre - half), min(1.0, centre + half))

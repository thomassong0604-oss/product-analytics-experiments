"""Estimating a treatment effect, and checking that the estimate is honest.

The modules split along that line. `estimate` computes effects and intervals, `power` says what the
experiment could ever have detected, `sequential` covers what repeated looks do to a p-value, and
`validate` runs the whole thing against data whose true effect is known and reports whether the
intervals actually cover it.
"""
from .estimate import (ClusterResult, Effect, cluster_bootstrap, cuped, delta_method_ratio,
                       naive_ratio, two_proportion, welch_mean)
from .power import design_effect, min_detectable_effect, sample_size_for_proportions
from .sequential import ALPHA_SPENDING, peeking_alpha, sequential_boundaries

__all__ = ["Effect", "ClusterResult", "two_proportion", "welch_mean", "naive_ratio",
           "delta_method_ratio", "cluster_bootstrap", "cuped", "sample_size_for_proportions",
           "min_detectable_effect", "design_effect", "sequential_boundaries", "peeking_alpha",
           "ALPHA_SPENDING"]

"""Power, and what repeated looks cost. The boundary values are checked against the literature."""
import pytest

from pae.experiment.power import (_z, design_effect, min_detectable_effect,
                                  sample_size_for_proportions)
from pae.experiment.sequential import _crossing_probability, peeking_alpha, sequential_boundaries


@pytest.mark.parametrize("level,expected", [(0.975, 1.959964), (0.80, 0.841621),
                                            (0.99, 2.326348), (0.999, 3.090232)])
def test_inverse_normal_is_accurate_in_the_tail(level, expected):
    """Every value this is asked for is a tail quantile. An approximation that is good in the middle
    and loose in the tail turns a 5% error rate into something else, which is a bug this project
    actually had."""
    assert _z(level) == pytest.approx(expected, abs=1e-5)


def test_sample_size_matches_the_textbook_formula():
    """Detecting a 10% relative lift on a 10% baseline at 80% power needs roughly 14,750 per arm."""
    result = sample_size_for_proportions(0.10, 0.10)
    assert 14_000 < result["n_control"] < 15_500
    assert result["n_control"] == result["n_treatment"]


def test_a_smaller_effect_needs_a_larger_sample():
    big = sample_size_for_proportions(0.10, 0.20)["n_total"]
    small = sample_size_for_proportions(0.10, 0.05)["n_total"]
    assert small > 3 * big        # requirement scales with the inverse square of the effect


def test_clustering_inflates_the_requirement_by_the_design_effect():
    plain = sample_size_for_proportions(0.10, 0.10)["n_total"]
    clustered = sample_size_for_proportions(0.10, 0.10, design=2.0)["n_total"]
    assert clustered == pytest.approx(2 * plain, rel=0.01)


def test_design_effect_formula():
    assert design_effect(1.0, 0.3) == pytest.approx(1.0)
    assert design_effect(2.6, 0.3) == pytest.approx(1.48)
    assert design_effect(8.0, 0.0) == pytest.approx(1.0)


def test_an_impossible_lift_is_refused_rather_than_silently_clipped():
    with pytest.raises(ValueError):
        sample_size_for_proportions(0.9, 0.5)          # takes the rate above 1
    with pytest.raises(ValueError):
        sample_size_for_proportions(0.1, 0.0)          # a zero lift needs infinite data


def test_detectable_effect_shrinks_with_the_square_root_of_the_sample():
    """Quadrupling the sample roughly halves the detectable effect.

    Roughly, not exactly: the variance under the alternative depends on the effect size itself, so
    the square-root law is an approximation that tightens as the effect gets small. Asserting it
    exactly would be asserting the approximation rather than the function.
    """
    small = min_detectable_effect(0.1, 1_000)["relative"]
    large = min_detectable_effect(0.1, 4_000)["relative"]
    assert large == pytest.approx(small / 2, rel=0.05)
    tiny = min_detectable_effect(0.1, 400_000)["relative"]
    tinier = min_detectable_effect(0.1, 1_600_000)["relative"]
    assert tinier == pytest.approx(tiny / 2, rel=0.005)   # the law tightens for small effects


def test_power_and_detectable_effect_are_consistent():
    """The sample size that gives 80% power for a 10% lift should report a 10% detectable effect."""
    n = sample_size_for_proportions(0.10, 0.10)["n_control"]
    assert min_detectable_effect(0.10, n)["relative"] == pytest.approx(0.10, rel=0.02)


@pytest.mark.parametrize("looks,published", [(2, 2.178), (3, 2.289), (5, 2.413)])
def test_pocock_boundaries_match_the_published_values(looks, published):
    """Reproduced from the numerical integration, not hardcoded — which is what makes the
    integration itself trustworthy."""
    assert sequential_boundaries(looks, 0.05, "pocock")[0] == pytest.approx(published, abs=0.003)


@pytest.mark.parametrize("looks,published", [(2, 1.977), (3, 2.004), (5, 2.040)])
def test_obrien_fleming_final_boundaries_match_the_published_values(looks, published):
    assert sequential_boundaries(looks, 0.05, "obrien-fleming")[-1] == pytest.approx(
        published, abs=0.003)


@pytest.mark.parametrize("looks", [1, 3, 8])
@pytest.mark.parametrize("family", ["pocock", "obrien-fleming"])
def test_calibrated_boundaries_hit_their_nominal_rate_exactly(looks, family):
    boundaries = sequential_boundaries(looks, 0.05, family)
    assert _crossing_probability(boundaries) == pytest.approx(0.05, abs=0.002)


def test_obrien_fleming_relaxes_and_pocock_does_not():
    obf = list(sequential_boundaries(6, 0.05, "obrien-fleming"))
    pocock = list(sequential_boundaries(6, 0.05, "pocock"))
    assert obf == sorted(obf, reverse=True)            # strict early, lenient late
    assert len(set(pocock)) == 1                       # one constant boundary
    assert obf[-1] < pocock[-1]                        # a cheaper final look


def test_peeking_inflates_the_false_positive_rate_and_the_boundaries_fix_it():
    result = peeking_alpha(10, iterations=20_000)
    assert result["naive_peeking"] > 0.15
    assert result["fixed_horizon"] == pytest.approx(0.05, abs=0.01)
    assert result["pocock"] == pytest.approx(0.05, abs=0.01)
    assert result["obrien_fleming"] == pytest.approx(0.05, abs=0.01)


def test_more_looks_cost_more():
    assert peeking_alpha(20, iterations=10_000)["naive_peeking"] > \
           peeking_alpha(3, iterations=10_000)["naive_peeking"]


def test_an_unknown_spending_function_is_refused():
    with pytest.raises(ValueError):
        sequential_boundaries(5, 0.05, "made-up")

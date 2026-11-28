import pytest
from estimates import chance_below, holm, zero_upper_bound
from scipy import stats


# Worked by hand: sorted, 0.005, 0.01, 0.03 and 0.04 are multiplied by 4, 3, 2 and 1,
# and 0.04 is raised to the 0.06 before it.
def test_holm_multiplies_by_the_tests_left_and_never_decreases():
    adjusted = holm([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


# The share of pairs where the first value is below the second, a tie counting half:
# a sample fully below gives 1, fully above 0, and an equal one 0.5.
@pytest.mark.parametrize(
    ("first", "second", "chance"),
    [
        ([1, 2], [3, 4], 1.0),
        ([3, 4], [1, 2], 0.0),
        ([1, 2], [1, 2], 0.5),
    ],
)
def test_chance_below_is_the_share_below_with_a_tie_as_half(first, second, chance):
    assert chance_below(first, second) == chance


# For zero hits the exact Clopper-Pearson upper bound is the 95% quantile of
# Beta(1, n).
def test_zero_upper_bound_is_the_clopper_pearson_bound():
    assert zero_upper_bound(666) == pytest.approx(stats.beta.ppf(0.95, 1, 666))

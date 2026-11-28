import pytest
from results import sign_test


# The upper-tailed example in https://www.stat.umn.edu/geyer/5601/examp/sign.html
# has the test statistic B = 21 for n = 25 and the P-value 0.00046.
@pytest.mark.parametrize(
    ("wins", "pairs", "p_value"),
    [
        (21, 25, 0.00046),
        (10, 10, 1 / 1024),
    ],
)
def test_sign_test_gives_the_p_value_of_at_least_this_many_wins(wins, pairs, p_value):
    assert sign_test(wins, pairs) == pytest.approx(p_value, abs=5e-6)


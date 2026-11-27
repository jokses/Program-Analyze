import pytest

from results import sign_test, split_and_domain


# The upper-tailed example in https://www.stat.umn.edu/geyer/5601/examp/sign.html
# has the test statistic B = 21 for n = 25 and the P-value 0.00046.
@pytest.mark.parametrize(
    ("wins", "n", "p"),
    [
        (21, 25, 0.00046),
        (10, 10, 1 / 1024),
    ],
)
def test_sign_test_gives_the_chance_of_at_least_this_many_wins(wins, n, p):
    assert sign_test(wins, n) == pytest.approx(p, abs=5e-6)


def test_split_and_domain_refuses_a_mix():
    three = {"split": "three", "domain": "sets"}
    assert split_and_domain([three, three]) == ("three", "sets")
    with pytest.raises(ValueError):
        split_and_domain([three, {"split": "boundary", "domain": "sets"}])

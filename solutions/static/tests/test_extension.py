import pytest
from absIntervals import INT_MAX, INT_MIN, Interval
from extension import CANDIDATES, candidates


@pytest.mark.parametrize(
    ("low", "high", "count"),
    [(0, 5, 6), (0, 1000, CANDIDATES), (INT_MIN, INT_MAX, CANDIDATES)],
)
def test_candidates_span_the_interval_with_at_most_candidates_values(low, high, count):
    values = candidates(Interval(low, high))
    assert (values[0], values[-1], len(values)) == (low, high, count)

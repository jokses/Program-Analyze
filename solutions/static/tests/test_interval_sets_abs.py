import pytest
from absIntervals import INT_MAX, INT_MIN, Interval
from absIntervalSets import IntervalSet

import jvm


def of(*pieces):
    return IntervalSet.make(Interval(low, high) for low, high in pieces)


def bounds(value):
    return [(piece.min, piece.max) for piece in value.pieces]


# A value that differs from a point leaves the point out of its middle.
@pytest.mark.parametrize(
    ("opr", "outcome"), [(jvm.CmpOpr.Ne, True), (jvm.CmpOpr.Eq, False)]
)
def test_refine_leaves_out_the_point_a_value_differs_from(opr, outcome):
    refined, _ = of((INT_MIN, INT_MAX)).refine(of((0, 0)), opr, outcome)
    assert bounds(refined) == [(INT_MIN, -1), (1, INT_MAX)]


# Java rounds a division toward 0, so 1 / -2 == 0.
def test_division_by_pieces_on_both_sides_of_zero_keeps_every_result():
    [value] = of((1, 1)).arithmetic(of((-2, -1), (1, 2)), jvm.BinaryOpr.Div)
    assert all(
        any(piece.min <= x <= piece.max for piece in value.pieces) for x in (-1, 0, 1)
    )


# i = i + 1 in a loop grows forever without widening.
def test_widening_stops_when_a_loop_counts_up():
    value = of((0, 0))
    for _ in range(20):
        [counted] = value.arithmetic(of((1, 1)), jvm.BinaryOpr.Add)
        widened = value.widen(value | counted, {0, 10})
        if widened == value:
            return
        value = widened
    pytest.fail("the widening did not stop")

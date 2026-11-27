import pytest
from absIntervals import INT_MAX, INT_MIN
from jvm_check import in_bound

import jvm
from jvm.state import PC

DIVIDE = PC(jvm.AbsMethodID.decode("jpamb.cases.Simple.checkBeforeDivideByN2:(I)I"), 4)
# n != 0 leaves out 0 from the divisor ranges, and operand 0 is not the divisor.
BOUND = {
    (DIVIDE, 1, INT_MIN, -1),
    (DIVIDE, 1, 1, INT_MAX),
    (DIVIDE, 0, 0, 0),
}


@pytest.mark.parametrize(
    ("value", "inside"),
    [(-5, True), (1, True), (INT_MAX, True), (0, False)],
)
def test_in_bound_only_with_a_divisor_range_around_the_value(value, inside):
    assert in_bound(BOUND, DIVIDE, value) == inside

import pytest
from jvm_check import covers

# The divisor ranges of checkBeforeDivideByN2, where n != 0 leaves out 0.
NOT_ZERO = [(-2147483648, -1), (1, 9), (10, 10), (11, 2147483647)]


@pytest.mark.parametrize(
    ("low", "high", "covered"),
    [
        (1, 11, True),
        (-5, -1, True),
        (0, 0, False),
        (-5, 9, False),
    ],
)
def test_covers_only_without_a_hole_from_low_to_high(low, high, covered):
    assert covers(NOT_ZERO, low, high) == covered

import jvm
import pytest

from dynamic import binary

INT_MIN = -(2**31)
INT_MAX = 2**31 - 1


@pytest.mark.parametrize(
    "operator, left, right, expected",
    [
        (jvm.BinaryOpr.Div, -7, 2, -3),
        (jvm.BinaryOpr.Div, INT_MIN, -1, INT_MIN),
        (jvm.BinaryOpr.Rem, -7, 2, -1),
        (jvm.BinaryOpr.Add, INT_MAX, 1, INT_MIN),
        (jvm.BinaryOpr.Div, 1, 0, "divide by zero"),
    ],
)
def test_binary_follows_java(operator, left, right, expected):
    assert binary(operator, left, right) == expected

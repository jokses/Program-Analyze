import pytest
from dynamic import binary, range_of, split_points

import jvm

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


@pytest.mark.parametrize(
    "split, value, expected",
    [
        ("three", 32, (32, 32)),
        ("three", 31, (1, 31)),
        ("three", 33, (33, INT_MAX)),
        ("three", -5, (INT_MIN, -1)),
        ("boundary", 31, (31, 31)),
        ("boundary", 30, (2, 30)),
        ("boundary", 34, (34, INT_MAX)),
        ("boundary", -1, (-1, -1)),
    ],
)
def test_range_of_splits_at_the_constants(split, value, expected):
    assert range_of(value, split_points({32}, split)) == expected

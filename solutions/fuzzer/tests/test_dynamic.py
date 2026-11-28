import pytest
from dynamic import (
    Draw,
    binary,
    bitwise,
    constant_code,
    fuzz_type,
    range_of,
    split_points,
)
from hypothesis import given
from hypothesis import strategies as st

import jpamb.case
import jvm

INT_MIN = -(2**31)
INT_MAX = 2**31 - 1


@pytest.mark.parametrize(
    ("operator", "left", "right", "expected"),
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
    ("operator", "left", "right", "expected"),
    [
        (jvm.BitOpr.Shl, 1, 33, 2),
        (jvm.BitOpr.Shl, 1, 31, INT_MIN),
        (jvm.BitOpr.Shr, -16, 2, -4),
        (jvm.BitOpr.Ushr, -1, 28, 15),
        (jvm.BitOpr.Ushr, -1, 0, -1),
    ],
)
def test_bitwise_follows_java(operator, left, right, expected):
    assert bitwise(operator, left, right) == expected


@pytest.mark.parametrize(
    ("split", "value", "expected"),
    [
        ("three", 32, (32, 32)),
        ("three", 31, (1, 31)),
        ("three", 33, (33, INT_MAX)),
        ("three", -5, (INT_MIN, -1)),
        ("boundary", 31, (31, 31)),
        ("boundary", 30, (2, 30)),
        ("boundary", -1, (-1, -1)),
    ],
)
def test_range_of_splits_at_the_constants(split, value, expected):
    assert range_of(value, split_points({32}, split)) == expected


PARAMETER_TYPES = st.sampled_from(
    [
        jvm.Int(),
        jvm.Boolean(),
        jvm.Array(jvm.Int()),
        jvm.Array(jvm.Char()),
        jvm.Object(jvm.ClassName("java.lang.String")),
    ]
)


@given(st.binary(), PARAMETER_TYPES)
def test_fuzz_type_gives_a_value_of_the_type(code: bytes, target: jvm.Type):
    value = fuzz_type(Draw(code), target)

    assert isinstance(value, jpamb.case.Value)
    assert value.type == target


@given(st.integers(INT_MIN, INT_MAX))
def test_constant_code_reads_back_as_the_constant(constant: int):
    params = [jvm.Int(), jvm.Array(jvm.Int())]
    draw = Draw(constant_code(params, constant))

    assert fuzz_type(draw, jvm.Int()).value == constant
    assert fuzz_type(draw, jvm.Array(jvm.Int())).values == (constant,)

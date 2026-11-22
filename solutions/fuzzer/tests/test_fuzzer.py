from dynamic import Draw, constant_code, fuzz_type
from hypothesis import given
from hypothesis import strategies as st

import jpamb.case
import jvm


def st_types():
    return st.sampled_from(
        [
            jvm.Int(),
            jvm.Boolean(),
            jvm.Array(jvm.Int()),
            jvm.Array(jvm.Char()),
            jvm.Object(jvm.ClassName("java.lang.String")),
        ]
    )


@given(st.binary(), st_types())
def test_fuzz_type(code: bytes, target: jvm.Type):
    value = fuzz_type(Draw(code), target)

    assert isinstance(value, jpamb.case.Value)
    assert value.type == target


@given(st.integers(-(2**31), 2**31 - 1))
def test_constant_code_reads_back_as_the_constant(constant: int):
    params = [jvm.Int(), jvm.Array(jvm.Int())]
    draw = Draw(constant_code(params, constant))

    assert fuzz_type(draw, jvm.Int()).value == constant
    assert fuzz_type(draw, jvm.Array(jvm.Int())).values == (constant,)

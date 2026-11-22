from dynamic import Draw, fuzz_type
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

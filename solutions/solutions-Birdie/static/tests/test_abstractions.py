import abstractions as ab
import pytest
from hypothesis import given
from hypothesis import strategies as st

import jvm
import jvm.state as jvms


def st_i32():
    return st.integers(min_value=-(2**31) - 1, max_value=2**31)


def st_u32():
    return st.integers(min_value=0, max_value=2**32)


def st_stack_ints():
    return st_i32().map(jvms.StackInt)


def st_stack_refs():
    return st_u32().map(jvms.StackReference)


def st_signset():
    return (
        st.sets(st.integers(min_value=-1, max_value=1)).map(frozenset).map(ab.SignSet)
    )


@given(st_signset(), st_signset(), st_signset())
def test_signset_is_poset(a: ab.SignSet, b: ab.SignSet, c: ab.SignSet):
    ab.is_poset(a, b, c)


@given(st_signset(), st_signset(), st_signset())
def test_signset_is_lattice(a: ab.SignSet, b: ab.SignSet, c: ab.SignSet):
    ab.is_lattice(a, b, c)


@given(st.sets(st_stack_ints()), st_signset())
def test_signset_is_galoi(a: set[jvms.StackInt], b: ab.SignSet):
    ab.is_galoi(a, b)


def arithmetic(opr, x, y):
    match opr:
        case jvm.BinaryOpr.Add:
            return jvms.StackInt(x.value + y.value)
        case jvm.BinaryOpr.Sub:
            return jvms.StackInt(x.value - y.value)
        case jvm.BinaryOpr.Mul:
            return jvms.StackInt(x.value * y.value)
        case jvm.BinaryOpr.Div:
            if y.value == 0:
                return None
            q = abs(x.value) // abs(y.value)
            return jvms.StackInt(q if (x.value < 0) == (y.value < 0) else -q)
        case jvm.BinaryOpr.Rem:
            if y.value == 0:
                return None
            remainder = abs(x.value) % abs(y.value)
            return jvms.StackInt(remainder if x.value >= 0 else -remainder)
        case _:
            raise NotImplementedError("TODO")


@given(
    st.sampled_from(jvm.BinaryOpr),
    st.sets(st_stack_ints()),
    st.sets(
        st_stack_ints(),
    ),
)
def test_signset_arithmetic(
    opr: jvm.BinaryOpr, xs: set[jvms.StackInt], ys: set[jvms.StackInt]
):
    results = [arithmetic(opr, x, y) for x in xs for y in ys]

    real = ab.SignSet.abstract(r for r in results if r is not None)
    outcomes = ab.SignSet.abstract(xs).arithmetic(ab.SignSet.abstract(ys), opr)
    abstracted = ab.SignSet.bot()
    errs = set()
    for outcome in outcomes:
        if isinstance(outcome, ab.SignSet):
            abstracted = abstracted | outcome
        else:
            errs.add(outcome)

    assert real <= abstracted
    if opr in {jvm.BinaryOpr.Div, jvm.BinaryOpr.Rem}:
        has_zero_divisor = bool(xs) and any(y.value == 0 for y in ys)
        assert ("divide by zero" in errs) == has_zero_divisor


def compare(opr, x, y):
    match opr:
        case jvm.CmpOpr.Le:
            return x.value <= y.value
        case jvm.CmpOpr.Lt:
            return x.value < y.value
        case jvm.CmpOpr.Ge:
            return x.value >= y.value
        case jvm.CmpOpr.Gt:
            return x.value > y.value
        case jvm.CmpOpr.Eq:
            return x.value == y.value
        case jvm.CmpOpr.Ne:
            return x.value != y.value
        case _:
            raise NotImplementedError("TODO")


@given(
    st.sampled_from(jvm.CmpOpr),
    st.sets(st_stack_ints()),
    st.sets(
        st_stack_ints(),
    ),
)
def test_signset_compare(
    opr: jvm.CmpOpr, xs: set[jvms.StackInt], ys: set[jvms.StackInt]
):
    real = {compare(opr, x, y) for x in xs for y in ys}
    abstracted = set(ab.SignSet.abstract(xs).compare(ab.SignSet.abstract(ys), opr))
    assert real <= abstracted


@st.composite
def st_interval(draw):
    if draw(st.booleans()):
        max = None
        if draw(st.booleans()):
            min = None
        else:
            min = draw(st_i32())
    elif draw(st.booleans()):
        min = None
        max = draw(st_i32())

    min, max = sorted([draw(st_i32()), draw(st_i32())])
    return ab.Interval(min, max)


@pytest.mark.skip("todo")
@given(st_interval(), st_interval(), st_interval())
def test_interval_is_poset(a: ab.Interval, b: ab.Interval, c: ab.Interval):
    ab.is_poset(a, b, c)


@pytest.mark.skip("todo")
@given(st_interval(), st_interval(), st_interval())
def test_interval_is_lattice(a: ab.Interval, b: ab.Interval, c: ab.Interval):
    ab.is_lattice(a, b, c)


@pytest.mark.skip("todo")
@given(st.sets(st_stack_ints()), st_interval())
def test_lnterval_is_galoi(a: set[jvms.StackInt], b: ab.Interval):
    ab.is_galoi(a, b)

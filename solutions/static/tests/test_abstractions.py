import abstractions as ab
import pytest
from hypothesis import given
from hypothesis import strategies as st

import jvm
import jvm.state as jvms


def st_i32():
    return st.integers(min_value=-(2**31), max_value=2**31 - 1)


def wrap32(v: int) -> int:
    return (v + 2**31) % 2**32 - 2**31


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
    """Concrete Java semantics. Returns None for division by zero."""
    a, b = x.value, y.value
    match opr:
        case jvm.BinaryOpr.Add:
            return jvms.StackInt(wrap32(a + b))
        case jvm.BinaryOpr.Sub:
            return jvms.StackInt(wrap32(a - b))
        case jvm.BinaryOpr.Mul:
            return jvms.StackInt(wrap32(a * b))
        case jvm.BinaryOpr.Div:
            if b == 0:
                return None
            q = abs(a) // abs(b)  # Java truncates toward zero,
            q = q if (a >= 0) == (b >= 0) else -q  # Python's // floors
            return jvms.StackInt(wrap32(q))
        case jvm.BinaryOpr.Rem:
            if b == 0:
                return None
            q = abs(a) // abs(b)
            q = q if (a >= 0) == (b >= 0) else -q
            return jvms.StackInt(wrap32(a - b * q))  # sign follows the dividend
        case _:
            raise NotImplementedError(f"TODO: {opr}")


@given(
    st.sampled_from(
        [
            jvm.BinaryOpr.Add,
            jvm.BinaryOpr.Sub,
            jvm.BinaryOpr.Mul,
            jvm.BinaryOpr.Div,
            jvm.BinaryOpr.Rem,
        ]
    ),
    st.sets(st_stack_ints()),
    st.sets(st_stack_ints()),
)
def test_signset_arithmetic(opr, xs, ys):
    concrete = [arithmetic(opr, x, y) for x in xs for y in ys]
    real = ab.SignSet.abstract(r for r in concrete if r is not None)

    results = ab.SignSet.abstract(xs).arithmetic(ab.SignSet.abstract(ys), opr)
    abstracted = ab.SignSet.bot()
    errs = set()
    for r in results:
        if isinstance(r, ab.SignSet):
            abstracted = abstracted | r
        else:
            errs.add(r)

    # Every value that can really occur must be in the abstract result
    assert real <= abstracted

    # Every real division by zero must be reported (extra warnings are fine)
    if any(r is None for r in concrete):
        assert "divide by zero" in errs


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

import absIntervals as ab
import operator

from tools import is_galoi, is_lattice, is_poset
import jvm.state as jvms

from hypothesis import given, settings, strategies as st


from absIntervals import INT_MAX, INT_MIN, Interval
import jvm

# Creating the overflow for the int range that Java handles
java_int = st.one_of(
    st.integers(-20, 20),
    st.integers(INT_MIN, INT_MIN + 5),
    st.integers(INT_MAX - 5, INT_MAX),
    st.integers(INT_MIN, INT_MAX),
)


def st_i32():
    return st.integers(min_value=-(2**31), max_value=2**31 - 1)


def st_stack_ints():
    return st_i32().map(jvms.StackInt)


@st.composite
def interval_with_member(draw):
    x, y = draw(java_int), draw(java_int)
    iv = Interval(min(x, y), max(x, y))
    v = draw(st.integers(iv.min, iv.max))
    return iv, v


intervals = st.one_of(
    st.just(Interval(1, 0)),
    st.just(Interval.top()),
    interval_with_member().map(lambda p: p[0]),
)


def wrap(v: int) -> int:
    return (v + 2**31) % 2**32 - 2**31


def java_div(x, y):
    q = abs(x) // abs(y)
    return wrap(q if (x < 0) == (y < 0) else -q)


def java_rem(x, y):
    return x - java_div(x, y) * y if (x, y) != (INT_MIN, -1) else 0


CONCRETE = {
    jvm.BinaryOpr.Add: lambda x, y: wrap(x + y),
    jvm.BinaryOpr.Sub: lambda x, y: wrap(x - y),
    jvm.BinaryOpr.Mul: lambda x, y: wrap(x * y),
    jvm.BinaryOpr.Div: java_div,
    jvm.BinaryOpr.Rem: java_rem,
}

CMP = {
    jvm.CmpOpr.Lt: operator.lt,
    jvm.CmpOpr.Le: operator.le,
    jvm.CmpOpr.Gt: operator.gt,
    jvm.CmpOpr.Ge: operator.ge,
    jvm.CmpOpr.Eq: operator.eq,
    jvm.CmpOpr.Ne: operator.ne,
}


@st.composite
def st_interval(draw):
    min, max = sorted([draw(st_i32()), draw(st_i32())])
    return ab.Interval(min, max)


@given(st_interval(), st_interval(), st_interval())
def test_interval_is_poset(a: ab.Interval, b: ab.Interval, c: ab.Interval):
    is_poset(a, b, c)


@given(st_interval(), st_interval(), st_interval())
def test_interval_is_lattice(a: ab.Interval, b: ab.Interval, c: ab.Interval):
    is_lattice(a, b, c)


@given(st.sets(st_stack_ints()), st_interval())
def test_interval_is_galoi(a: set[jvms.StackInt], b: ab.Interval):
    is_galoi(a, b)


def test_bot_is_unique():
    assert Interval(5, 2) == Interval(1, 0) == Interval.bot()


@settings(max_examples=500)
@given(
    interval_with_member(),
    interval_with_member(),
    st.sampled_from(list(CONCRETE)),
)
def test_arithmetic_sound(p, q, opr):
    (a, x), (b, y) = p, q
    results = a.arithmetic(b, opr)
    if opr in (jvm.BinaryOpr.Div, jvm.BinaryOpr.Rem) and y == 0:
        assert "divide by zero" in results
        return
    r = CONCRETE[opr](x, y)
    assert any(
        isinstance(i, Interval) and jvms.StackInt(r) in i for i in results
    ), f"{x} {opr} {y} = {r} not in {results} for {a}, {b}"


@settings(max_examples=500)
@given(interval_with_member(), interval_with_member(), st.sampled_from(list(CMP)))
def test_compare_and_refine_sound(p, q, opr):
    (a, x), (b, y) = p, q
    outcome = CMP[opr](x, y)
    assert outcome in a.compare(b, opr)
    na, nb = a.refine(b, opr, outcome)
    assert jvms.StackInt(x) in na and jvms.StackInt(y) in nb


@given(intervals, intervals, st.sets(st.integers(-100, 100), max_size=5))
def test_widen_is_upper_bound(a, b, K):
    w = a.widen(b, K)
    assert a <= w and b <= w


@given(
    st.lists(intervals, min_size=1, max_size=200),
    st.sets(st.integers(-100, 100), max_size=5),
)
def test_widen_stabilises(chain, K):
    acc = Interval(1, 0)
    changes = 0
    for iv in chain:
        nxt = acc.widen(acc | iv, K)
        changes += nxt != acc
        acc = nxt
    assert changes <= 2 * (len(K) + 2) + 1


@given(interval_with_member(), st.sets(st.integers(-100, 100), max_size=5))
def test_split_partitions(p, K):
    iv, v = p
    parts = iv.split(K)
    assert sum(jvms.StackInt(v) in c for c in parts) == 1
    for k in K:
        if iv.min <= k <= iv.max:
            assert Interval(k, k) in parts


def test_division_loop_example():
    assert [str(c) for c in Interval.top().split({32})] == [
        f"[{INT_MIN}, 31]",
        "[32, 32]",
        f"[33, {INT_MAX}]",
    ]

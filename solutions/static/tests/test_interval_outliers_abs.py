"""Tests for IntervalOutliers.

Run with:  python test_interval_outliers.py

Each test runs on its own, so one failure doesn't stop the others.
At the end you get a list of which tests passed and failed.

Change the import below if your file has a different name.
"""

import random
import signal
import sys
import traceback

import absOutliers as mod
from absIntervals import Interval
from absOutliers import IntervalOutliers as IO

import jvm
from jvm.state import StackInt as SV

B, C = jvm.BinaryOpr, jvm.CmpOpr
GAP, MAX_OUTLIERS = mod.GAP, mod.MAX_OUTLIERS


def val(*ints):
    """Abstract value of the given ints, e.g. val(0, 1, 2, 3, 4, 1234)."""
    return IO.abstract([SV(x) for x in ints])


def ints_of(v):
    """All ints in a small abstract value (only use on small values!)."""
    out = set(v.outliers)
    if not v.iv.is_bot():
        out |= set(range(v.iv.min, v.iv.max + 1))
    return out


def has(v, x):
    return SV(x) in v


def java(x, y, op):
    """What Java computes for x op y on ints."""
    wrap = lambda n: (n + 2**31) % 2**32 - 2**31
    if op == B.Add:
        return wrap(x + y)
    if op == B.Sub:
        return wrap(x - y)
    if op == B.Mul:
        return wrap(x * y)
    if y == 0:
        return "divide by zero"
    q = abs(x) // abs(y)
    q = q if (x >= 0) == (y >= 0) else -q  # Java truncates toward zero
    return wrap(q) if op == B.Div else wrap(x - q * y)


def java_cmp(x, y, op):
    return {
        C.Le: x <= y,
        C.Lt: x < y,
        C.Ge: x >= y,
        C.Gt: x > y,
        C.Eq: x == y,
        C.Ne: x != y,
    }[op]


def result_has(res, r):
    """Does the result list of arithmetic contain the concrete result r?"""
    if isinstance(r, str):
        return r in res
    return any(not isinstance(v, str) and has(v, r) for v in res)


def test_make_keeps_far_point_as_outlier():
    v = val(0, 1, 2, 3, 4, 1234)
    assert v.iv == Interval(0, 4), v
    assert v.outliers == frozenset({1234}), v


def test_make_merges_close_point():
    v = val(0, 4, 6)
    assert v.iv == Interval(0, 6), v
    assert v.outliers == frozenset(), v


def test_make_returns_a_value():
    v = IO.make([Interval(0, 4)])
    assert isinstance(v, IO), f"make returned {v!r}"


def test_make_absorbs_point_inside_interval():
    v = IO.make([Interval(0, 100), Interval(50, 50)])
    assert v.iv == Interval(0, 100) and not v.outliers, v


def test_make_absorbs_point_close_after_hull():
    # 30 is far from [0,4] but close to [20,25]; after hulling it must be absorbed
    v = IO.make([Interval(0, 4), Interval(20, 25), Interval(30, 30)])
    assert v.iv == Interval(0, 30) and not v.outliers, v


def test_make_respects_max_outliers():
    v = val(*[1000 * i for i in range(1, MAX_OUTLIERS + 5)])
    assert len(v.outliers) <= MAX_OUTLIERS, v


def test_make_never_loses_values():
    xs = [1000 * i for i in range(1, MAX_OUTLIERS + 5)] + [0, 1, 2]
    v = val(*xs)
    for x in xs:
        assert has(v, x), (x, v)


def test_outliers_are_ints():
    v = val(0, 1, 2, 3, 4, 1234)
    for o in v.outliers:
        assert isinstance(o, int), f"outlier {o!r} is not an int"


def test_no_outlier_inside_interval():
    for _ in range(200):
        v = val(*random.sample(range(-200, 200), 6))
        for o in v.outliers:
            assert not (v.iv.min <= o <= v.iv.max), v


def test_bot_and_top():
    assert IO.bot().is_bot()
    assert not IO.top().is_bot()
    assert not IO(Interval.bot(), frozenset({5})).is_bot()


def test_contains():
    v = val(0, 1, 2, 3, 4, 1234)
    assert has(v, 2) and has(v, 1234)
    assert not has(v, 500)


def test_str():
    assert str(IO.bot()) == "⊥"
    s = str(IO(Interval.bot(), frozenset({7})))
    assert "7" in s and "⊥" not in s, f"points only printed as {s!r}"
    s = str(val(0, 1, 2, 3, 4, 1234))
    assert "[0, 4]" in s and "1234" in s, s


def test_join():
    v = val(0, 1, 2, 3, 4) | val(1234)
    assert v.iv == Interval(0, 4) and v.outliers == frozenset({1234}), v


def test_meet():
    v = val(0, 1, 2, 3, 4, 1234) & val(3, 4, 5, 6, 1234)
    assert has(v, 3) and has(v, 4) and has(v, 1234), v
    assert not has(v, 0), v


def test_le_table():
    cases = [
        (val(0, 1, 2, 3, 4, 500), val(*range(101), 1000), False),
        (val(*range(200, 205)), val(*range(101), 1000), False),
        (val(1000), val(*range(101), 1000), True),
        (val(*range(101), 1000), val(*range(101), 1000), True),
        (IO(Interval(5, 6)), IO(Interval(0, 4), frozenset({5, 6})), True),
        (IO.bot(), val(*range(101), 1000), True),
        (val(*range(11)), IO(Interval(0, 4), frozenset({10})), False),
    ]
    for a, b, expected in cases:
        assert (a <= b) == expected, f"{a} <= {b} gave {a <= b}, expected {expected}"


def test_lt():
    a, b = val(1, 2, 3), val(*range(10))
    assert a < b
    assert not b < a
    assert not a < a


def test_le_with_top_is_fast():
    # Fails (times out) until you do step C, the "leftovers" trick.
    assert IO.top() <= IO.top()
    assert val(5) <= IO.top()


def test_arithmetic_add():
    res = val(0, 1, 2, 3, 4, 1234).arithmetic(val(1), B.Add)
    vals = [r for r in res if not isinstance(r, str)]
    assert len(vals) == 1, res
    v = vals[0]
    assert isinstance(v, IO), f"result is {type(v).__name__}, not IntervalOutliers"
    assert v.iv == Interval(1, 5) and v.outliers == frozenset({1235}), v


def test_arithmetic_div_by_zero_only():
    res = val(10, 20).arithmetic(val(0), B.Div)
    assert res == ["divide by zero"], res


def test_arithmetic_error_once():
    res = val(0, 1, 2, 3, 4, 1234).arithmetic(IO(Interval(-1, 5)), B.Div)
    assert res.count("divide by zero") == 1, res


def test_arithmetic_bot():
    res = IO.bot().arithmetic(val(1), B.Add)
    assert all(isinstance(r, IO) and r.is_bot() for r in res), res


def test_compare_uses_holes():
    # plain intervals would say {True, False}
    assert set(val(0, 1, 2, 3, 4, 1234).compare(val(32), C.Eq)) == {False}


def test_refine():
    a, b = val(0, 1, 2, 3, 4, 1234), val(100)
    na, _ = a.refine(b, C.Gt, True)  # x > 100
    assert ints_of(na) == {1234}, na
    na, _ = a.refine(b, C.Gt, False)  # x <= 100
    assert ints_of(na) == {0, 1, 2, 3, 4}, na


def test_widen_upper_bound():
    K = {0, 10, 10000}
    a = val(0, 1, 2, 3, 4, 1234)
    b = val(0, 1, 2, 3, 4, 5, 1234, 5000)
    w = a.widen(b, K)
    assert a <= w and b <= w, w


def test_widen_adds_no_new_outliers():
    K = {0, 100}
    a = val(0, 1, 2, 3, 4, 1234)
    b = val(0, 1, 2, 3, 4, 1234, 5000)
    w = a.widen(b, K)
    assert w.outliers <= a.outliers, f"new outliers appeared: {w}"


def test_widen_terminates():
    K = {0, 100}
    x = val(0)
    for n in range(1, 200):
        nxt = x | val(10 * n * n) | x.arithmetic(val(1), B.Add)[0]
        w = x.widen(nxt, K)
        if w == x:
            return
        x = w
    raise AssertionError(f"widening did not stabilise after 200 steps: {x}")


def test_split_is_partition():
    K = {0, 2, 50}
    v = val(0, 1, 2, 3, 4, 1234)
    ranges = v.split(K)
    for x in ints_of(v):
        hits = sum(r.min <= x <= r.max for r in ranges)
        assert hits == 1, f"{x} is in {hits} ranges: {[str(r) for r in ranges]}"
    covered = set().union(*(range(r.min, r.max + 1) for r in ranges))
    assert covered == ints_of(v), ranges


def test_random_soundness():
    """Random small sets: every concrete Java result must be in the abstract result."""
    rnd = random.Random(1)
    for _ in range(500):
        X = [
            rnd.choice([rnd.randint(-20, 20), rnd.randint(-3000, 3000)])
            for _ in range(rnd.randint(1, 5))
        ]
        Y = [
            rnd.choice([rnd.randint(-20, 20), rnd.randint(-3000, 3000)])
            for _ in range(rnd.randint(1, 5))
        ]
        a, b = val(*X), val(*Y)
        for op in B:
            res = a.arithmetic(b, op)
            for x in X:
                for y in Y:
                    assert result_has(res, java(x, y, op)), (str(a), op, str(b), x, y)
        for op in [C.Le, C.Lt, C.Ge, C.Gt, C.Eq, C.Ne]:
            cs = set(a.compare(b, op))
            for outcome in (True, False):
                na, nb = a.refine(b, op, outcome)
                for x in X:
                    for y in Y:
                        if java_cmp(x, y, op) == outcome:
                            assert outcome in cs, ("compare", str(a), op, str(b), x, y)
                            assert has(na, x) and has(nb, y), (
                                "refine",
                                str(a),
                                op,
                                str(b),
                                x,
                                y,
                                outcome,
                            )
        j = a | b
        assert all(has(j, x) for x in X + Y), ("join", str(a), str(b))


TIMEOUT = 5  # seconds per test (catches infinite loops; only works on Linux/macOS)


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout(f"took more than {TIMEOUT}s (infinite loop?)")


if __name__ == "__main__":
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        signal.signal(signal.SIGALRM, _alarm)
    passed = 0
    for name, f in tests:
        try:
            if use_alarm:
                signal.alarm(TIMEOUT)
            f()
            print(f"  ok    {name}")
            passed += 1
        except Exception as e:
            kind = type(e).__name__
            msg = str(e).splitlines()[0][:150] if str(e) else ""
            print(f"  FAIL  {name}: {kind}: {msg}")
            if "-v" in sys.argv:
                traceback.print_exc()
        finally:
            if use_alarm:
                signal.alarm(0)
    print(f"\n{passed}/{len(tests)} passed")

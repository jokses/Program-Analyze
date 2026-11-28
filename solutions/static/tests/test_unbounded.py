from pathlib import Path

import dynamic
import fuzzer
import pytest
from unbounded import (
    INT_MIN,
    Interval,
    RangeCoverageOracle,
    State,
    Val,
    cast_interval,
    map_pieces,
    negate_interval,
    pieces_from_interval,
)

import jpamb
import jvm
from jvm.state import PC

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def bc():
    return jpamb.Bytecode(jpamb.Suite.from_workdir(REPO), jpamb.Effect(None), {})


def test_split_gives_a_piece_per_point_and_between_points():
    pieces = Interval(-5, 400).split([0, 300])
    assert [(piece.min, piece.max) for piece in pieces] == [
        (-5, -1),
        (0, 0),
        (1, 299),
        (300, 300),
        (301, 400),
    ]


def test_forget_elements_forgets_the_elements_of_a_passed_array():
    array = Val(Interval.const(3), elem_iv=Interval.const(0), alloc_id=4, is_ref=True)
    state = State((array,), ())
    passed = Val(
        Interval.const(3), src=0, elem_iv=Interval.const(0), alloc_id=4, is_ref=True
    )
    assert state.forget_elements([passed]).locals[0].elem_iv == Interval.top()


# n != 0 is checked before 1 / n so n (operand 1) is never 0 at the division.
# The interval sets and the signs can leave out 0 inside an interval and the
# outliers cannot.
@pytest.mark.parametrize(
    ("domain", "zero_in_bound"),
    [("sets", False), ("signs", False), ("outliers", True)],
)
def test_build_leaves_out_zero_at_the_division_after_n_not_zero(
    bc, domain, zero_in_bound
):
    methodid = jvm.AbsMethodID.decode("jpamb.cases.Simple.checkBeforeDivideByN2:(I)I")
    oracle = RangeCoverageOracle.build(bc, methodid, "three", domain)
    assert ((PC(methodid, 4), 1, 0, 0) in oracle.upper_bound) == zero_in_bound


def test_solve_uncovered_range_undoes_the_loop_in_division_loop(bc):
    methodid = jvm.AbsMethodID.decode("jpamb.cases.Dependent.divisionLoop:(I)V")
    oracle = RangeCoverageOracle.build(bc, methodid, "three")
    input = jpamb.case.Input((jpamb.case.Int(7),))
    run = dynamic.run(bc, methodid, input, max_steps=fuzzer.MAX_STEPS)
    # n == 32 at the assertion, after dividing n by 2 five times.
    target = (PC(methodid, 15), 0, 32, 32)
    assert oracle.analysis.solve_uncovered_range(target, run.path) == Interval(
        1024, 1055
    )


def to_byte(iv):
    return cast_interval(iv, Interval(-128, 127))


def to_char(iv):
    return cast_interval(iv, Interval(0, 65535))


# The value Java gives has to be in the pieces after the operation.
@pytest.mark.parametrize(
    ("operation", "low", "high", "java"),
    [
        (negate_interval, INT_MIN, INT_MIN, INT_MIN),
        (to_byte, 0, 300, 127),
        (to_char, -1, 5, 0),
    ],
)
def test_map_pieces_keeps_the_value_java_gives(operation, low, high, java):
    out = map_pieces(pieces_from_interval(Interval(low, high)), operation)
    assert any(piece.min <= java <= piece.max for piece in out.pieces)

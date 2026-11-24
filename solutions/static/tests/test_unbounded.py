from pathlib import Path

import dynamic
from unbounded import Interval, RangeCoverageOracle, State, Val

import jpamb
import jvm
from jvm.state import PC

REPO = Path(__file__).resolve().parents[3]


def test_split_gives_a_piece_per_point_and_between_points():
    pieces = Interval(-5, 400).split([0, 300])
    assert [(piece.min, piece.max) for piece in pieces] == [
        (-5, -1),
        (0, 0),
        (1, 299),
        (300, 300),
        (301, 400),
    ]


def test_call_forgets_the_elements_of_a_passed_array():
    array = Val(Interval.const(3), elem_iv=Interval.const(0), alloc_id=4, is_ref=True)
    state = State((array,), ())
    passed = Val(
        Interval.const(3), src=0, elem_iv=Interval.const(0), alloc_id=4, is_ref=True
    )
    assert state.forget_elements([passed]).locals[0].elem_iv == Interval.top()


def test_backwards_through_the_loop_in_division_loop():
    suite = jpamb.Suite.from_workdir(REPO)
    bc = jpamb.Bytecode(suite, jpamb.Effect(None), {})
    methodid = jvm.AbsMethodID.decode("jpamb.cases.Dependent.divisionLoop:(I)V")
    oracle = RangeCoverageOracle.build(bc, methodid, "three")
    run = dynamic.run(
        bc, methodid, jpamb.case.Input((jpamb.case.Int(7),)), max_steps=1000
    )
    # n == 32 at the assertion, after dividing n by 2 five times.
    target = (PC(methodid, 15), 0, 32, 32)
    assert oracle.analysis.solve_uncovered_range(target, run.path) == Interval(
        1024, 1055
    )

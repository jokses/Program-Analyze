from unbounded import Interval, State, Val


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

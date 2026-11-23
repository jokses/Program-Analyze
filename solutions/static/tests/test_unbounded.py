from unbounded import Interval


def test_split_gives_a_piece_per_point_and_between_points():
    pieces = Interval(-5, 400).split([0, 300])
    assert [(piece.min, piece.max) for piece in pieces] == [
        (-5, -1),
        (0, 0),
        (1, 299),
        (300, 300),
        (301, 400),
    ]

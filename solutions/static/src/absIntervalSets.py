import bisect
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

from absIntervals import INT_MAX, INT_MIN, Interval
from tools import Abstraction, Lattice

import jvm
import jvm.state as jvms


# A value is a set of separate intervals. One interval cannot say n != 0 on all
# ints, but [INT_MIN, -1] and [1, INT_MAX] can. The sign set {-, +} says it too,
# and this keeps an interval for each part instead of only its sign.
# Source: https://courses.compute.dtu.dk/02242/topics/bounded-static-analysis.html#sec:2.1
# Each operation goes piece by piece. That is sound because Java's operations act
# on each value on its own, like the powerset domain in Definition 3 of Milanese
# and Mine (VMCAI 2024).
# Source: https://hal.sorbonne-universite.fr/hal-04317611v1
@dataclass(frozen=True)
class IntervalSet(Abstraction, Lattice):
    intervals: tuple[Interval, ...] = ()

    # Overlapping and touching pieces are merged so equal sets look the same.
    @classmethod
    def make(cls, pieces: Iterable[Interval]) -> Self:
        merged: list[Interval] = []
        for piece in sorted(
            (piece for piece in pieces if not piece.is_bot()),
            key=lambda piece: piece.min,
        ):
            if merged and piece.min <= merged[-1].max + 1:
                merged[-1] = merged[-1] | piece
            else:
                merged.append(piece)
        return cls(tuple(merged))

    def pieces(self) -> list[Interval]:
        return list(self.intervals)

    @classmethod
    def bot(cls) -> Self:
        return cls()

    @classmethod
    def top(cls) -> Self:
        return cls((Interval.top(),))

    @classmethod
    def abstract(cls, values: Iterable[jvms.StackValue]) -> Self:
        return cls.make(Interval(value.value, value.value) for value in values)

    def __contains__(self, value: jvms.StackValue) -> bool:
        return any(value in piece for piece in self.intervals)

    def is_bot(self) -> bool:
        return not self.intervals

    def __or__(self, other: Self) -> Self:
        return self.make(self.pieces() + other.pieces())

    def __and__(self, other: Self) -> Self:
        return self.make(
            left & right for left in self.intervals for right in other.intervals
        )

    # The pieces are merged so each piece of self has to lie in one piece of other.
    def __le__(self, other: Self) -> bool:
        return all(
            any(
                right.min <= left.min and left.max <= right.max
                for right in other.intervals
            )
            for left in self.intervals
        )

    def __lt__(self, other: Self) -> bool:
        return self <= other and self != other

    def __gt__(self, other: Self) -> bool:
        return other < self

    def __str__(self) -> str:
        return " | ".join(map(str, self.intervals)) or "bot"

    def arithmetic(self, other: Self, opr: jvm.BinaryOpr) -> list[Self | str]:
        if self.is_bot() or other.is_bot():
            return [self.bot()]
        errors: list[str] = []
        values: list[Interval] = []
        for left in self.intervals:
            for right in other.intervals:
                for result in left.arithmetic(right, opr):
                    if isinstance(result, str):
                        if result not in errors:
                            errors.append(result)
                    else:
                        values.append(result)
        return [*errors, self.make(values)] if values else list(errors)

    def compare(self, other: Self, opr: jvm.CmpOpr) -> set[bool]:
        return {
            outcome
            for left in self.intervals
            for right in other.intervals
            for outcome in left.compare(right, opr)
        }

    # Interval.refine can only cut a point off an end, and this also cuts a point
    # in the middle, like 0 out of all ints for n != 0.
    def refine(self, other: Self, opr: jvm.CmpOpr, outcome: bool) -> tuple[Self, Self]:
        differs = (opr, outcome) in {(jvm.CmpOpr.Ne, True), (jvm.CmpOpr.Eq, False)}
        self_kept: list[Interval] = []
        other_kept: list[Interval] = []
        for left in self.intervals:
            for right in other.intervals:
                if outcome not in left.compare(right, opr):
                    continue
                left_kept, right_kept = left.refine(right, opr, outcome)
                self_kept += without(left_kept, right) if differs else [left_kept]
                other_kept += without(right_kept, left) if differs else [right_kept]
        return self.make(self_kept), self.make(other_kept)

    # Each range between the constants in K is widened on its own and stays inside
    # it. There are finitely many ranges and each can only grow to the whole range
    # so the widening stops.
    def widen(self, other: Self, K: set) -> Self:
        constants = sorted(K)
        old, new = by_range(self, constants), by_range(other, constants)
        return self.make(
            old.get(bounds, Interval.bot()).widen(new.get(bounds, Interval.bot()), K)
            & Interval(*bounds)
            for bounds in old.keys() | new.keys()
        )


def without(iv: Interval, point: Interval) -> list[Interval]:
    if point.min != point.max or not iv.min < point.min < iv.max:
        return [iv]
    return [Interval(iv.min, point.min - 1), Interval(point.min + 1, iv.max)]


# The range of x is [k, k] for a constant k, else the ints between two constants.
def range_around(x: int, constants: list[int]) -> tuple[int, int]:
    i = bisect.bisect_left(constants, x)
    if i < len(constants) and constants[i] == x:
        return x, x
    low = constants[i - 1] + 1 if i > 0 else INT_MIN
    high = constants[i] - 1 if i < len(constants) else INT_MAX
    return low, high


def by_range(
    value: IntervalSet, constants: list[int]
) -> dict[tuple[int, int], Interval]:
    ranges: dict[tuple[int, int], Interval] = {}
    for piece in value.intervals:
        for sub in piece.split(constants):
            bounds = range_around(sub.min, constants)
            ranges[bounds] = ranges.get(bounds, Interval.bot()) | sub
    return ranges

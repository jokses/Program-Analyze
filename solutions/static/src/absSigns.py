from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Self

from absIntervals import INT_MAX, INT_MIN, Interval
from tools import Abstraction, Lattice

import jvm
import jvm.state as jvms

type Sign = Literal[-1, 0, 1]

# The ints each sign stands for.
SIGN_INTERVALS = {-1: Interval(INT_MIN, -1), 0: Interval(0, 0), 1: Interval(1, INT_MAX)}


def to_sign(value: jvms.StackValue) -> Sign:
    match value:
        case jvms.StackInt(v):
            return (v > 0) - (v < 0)
        case a:
            raise NotImplementedError(f"Unsupported value {value!r}")


@dataclass(frozen=True)
class SignSet(Abstraction, Lattice):
    signs: frozenset[Sign]

    def __lt__(self, other: "SignSet") -> bool:
        return self.signs < other.signs

    def __gt__(self, other: "SignSet") -> bool:
        return self.signs > other.signs

    def __str__(self):
        return f"{{{''.join(z for z, b in zip('-0+', [-1, 0, 1]) if b in self.signs)}}}"

    def __sexpr__(self):
        return str(self)

    @classmethod
    def from_sign(cls, code):
        return cls(frozenset(b for z, b in zip("-0+", [-1, 0, 1]) if z in code))

    @classmethod
    def abstract(cls, values: Iterable[jvms.StackValue]) -> Self:
        signs = set()
        for value in values:
            signs.add(to_sign(value))
        return cls(frozenset(signs))

    def __contains__(self, value: jvms.StackValue) -> bool:
        return to_sign(value) in self.signs

    @classmethod
    def top(cls) -> Self:
        return cls(frozenset([-1, 0, 1]))

    # The analysis keeps signs next to its intervals as pieces, one interval per sign.
    @classmethod
    def make(cls, pieces: Iterable[Interval]) -> Self:
        return cls(
            frozenset(
                sign
                for piece in pieces
                for sign, ints in SIGN_INTERVALS.items()
                if not (piece & ints).is_bot()
            )
        )

    @property
    def pieces(self) -> list[Interval]:
        return [SIGN_INTERVALS[sign] for sign in sorted(self.signs)]

    def refine(
        self, other: "SignSet", opr: jvm.CmpOpr, outcome: bool
    ) -> tuple["SignSet", "SignSet"]:
        self_kept: list[Interval] = []
        other_kept: list[Interval] = []
        for left in self.pieces:
            for right in other.pieces:
                if outcome in left.compare(right, opr):
                    left_kept, right_kept = left.refine(right, opr, outcome)
                    self_kept.append(left_kept)
                    other_kept.append(right_kept)
        return self.make(self_kept), self.make(other_kept)

    # There are only eight sign sets, so joining stops by itself.
    def widen(self, other: "SignSet", K: set) -> "SignSet":
        return self | other

    def __or__(self, other: "SignSet") -> "SignSet":
        if not isinstance(other, SignSet):
            return NotImplemented

        return SignSet(self.signs | other.signs)

    def __and__(self, other: "SignSet") -> "SignSet":
        if not isinstance(other, SignSet):
            return NotImplemented

        return SignSet(self.signs & other.signs)

    # The following operations are not very specific in many of the outputs, often
    # being [-1, 0, 1] where one might consider it irrelevant. However, this is because
    # of the range of integers. Java transforms them like in a circle meaning adding
    # two integers will become negative if exceeding this range.
    def arithmetic(self, other: "SignSet", opr: jvm.BinaryOpr) -> list["SignSet | str"]:
        match opr:
            case jvm.BinaryOpr.Add:
                output = set()
                if 1 in self.signs:
                    output.update([1, -1])
                    if -1 in other.signs:
                        output.update([0, -1])

                if -1 in self.signs:
                    output.add(-1)
                    # Two negatives can wrap too, like -1 + INT_MIN == INT_MAX.
                    if 1 in other.signs or -1 in other.signs:
                        output.update([0, 1])

                if 0 in self.signs:
                    output.update(other.signs)

                return [SignSet(output)] if output else []

            case jvm.BinaryOpr.Sub:
                output = set()

                for x in self.signs:
                    for y in other.signs:
                        if x == 0 and y == 0:
                            output.add(0)
                            continue
                        elif x == y:
                            output.update([-1, 0, 1])
                        elif x > y:
                            output.update([1, -1])
                        else:
                            output.update([1, -1])
                return [SignSet(output)] if output else []

            case jvm.BinaryOpr.Mul:
                output = set()

                for x in self.signs:
                    for y in other.signs:
                        if x == 0 or y == 0:
                            output.add(0)
                            continue
                        elif (x == -1) != (y == -1):
                            output.update([-1, 0, 1])
                        else:
                            output.update([-1, 0, 1])

                return [SignSet(output)] if output else []

            case jvm.BinaryOpr.Div:
                output = set()
                errors = set()

                for numerator in self.signs:
                    for denominator in other.signs:
                        if denominator == 0:
                            errors.add("divide by zero")
                        elif numerator == 0:
                            output.add(0)
                        elif numerator == denominator:
                            output.update([-1, 0, 1])
                        else:
                            output.update([0, -1])

                results = [SignSet(output)] if output else []
                return [*results, *sorted(errors)]

            case jvm.BinaryOpr.Rem:
                output = set()
                errors = set()

                for x in self.signs:
                    for y in other.signs:
                        if y == 0:
                            errors.add("divide by zero")
                        elif x == 0:
                            output.add(0)
                        elif x > 0:
                            output.update([0, 1])
                        else:
                            output.update([-1, 0])

                results = [SignSet(output)] if output else []
                return [*results, *sorted(errors)]

            case _:
                # print("look here", opr, file=sys.stderr)
                raise NotImplementedError(f"TODO: {opr}")

    def compare(self, other: "SignSet", opr: jvm.CmpOpr) -> Iterable[bool]:
        match opr:
            case jvm.CmpOpr.Le:
                cases = set()
                for x in self.signs:
                    for y in other.signs:
                        if x == 0 or y == 0:
                            cases.add(x <= y)
                            continue
                        if x <= y:
                            cases.add(True)
                        if x >= y:
                            cases.add(False)
                return cases
            case jvm.CmpOpr.Lt:
                cases = set()
                for x in self.signs:
                    for y in other.signs:
                        if x == 0 or y == 0:
                            cases.add(x < y)
                            continue
                        if x <= y:
                            cases.add(True)
                        if x >= y:
                            cases.add(False)
                return cases
            case jvm.CmpOpr.Ge:
                cases = set()
                for x in self.signs:
                    for y in other.signs:
                        if x == 0 or y == 0:
                            cases.add(x >= y)
                            continue
                        if x >= y:
                            cases.add(True)
                        if x <= y:
                            cases.add(False)
                return cases
            case jvm.CmpOpr.Gt:
                cases = set()
                for x in self.signs:
                    for y in other.signs:
                        if x == 0 or y == 0:
                            cases.add(x > y)
                            continue
                        if x >= y:
                            cases.add(True)
                        if x <= y:
                            cases.add(False)
                return cases
            case jvm.CmpOpr.Eq:
                cases = set()
                for x in self.signs:
                    for y in other.signs:
                        if x == 0 and y == 0:
                            cases.add(True)
                            continue
                        if x == y:
                            cases.update([True, False])
                        if x != y:
                            cases.add(False)
                return cases
            case jvm.CmpOpr.Ne:
                cases = set()
                for x in self.signs:
                    for y in other.signs:
                        if x != y:
                            cases.add(True)
                        elif x == 0 and y == 0:
                            cases.add(False)
                        else:
                            cases.update([True, False])
                return cases
            case _:
                raise NotImplementedError(f"TODO: {opr}")

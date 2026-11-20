from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

from tools import Abstraction, Lattice

import jvm
import jvm.state as jvms

INT_MIN = -(2**31)
INT_MAX = (2**31) - 1


# this method will handle if overflow for the java integers should happen.
# It selects either the very top of the interval or it wraps which circles the interval
def wrap_or_top(lo, hi) -> "Interval":
    if lo < INT_MIN or hi > INT_MAX:
        return Interval(INT_MIN, INT_MAX)
    return Interval(lo, hi)


@dataclass(frozen=True)
class Interval(Abstraction, Lattice):
    min: int
    max: int

    def __post_init__(self):
        if self.min > self.max:
            object.__setattr__(self, "min", 1)
            object.__setattr__(self, "max", 0)

    @classmethod
    def bot(cls):
        return cls(1, 0)

    def is_bot(self) -> bool:
        return self.min > self.max

    def __str__(self):
        return "⊥" if self.is_bot() else f"[{self.min}, {self.max}]"

    def __sexpr__(self):
        return str(self)

    @classmethod
    def abstract(cls, values: Iterable[jvms.StackValue]) -> Self:
        nums = [v.value for v in values]

        if not nums:
            # The interval [1, 0] is empty as there is no numbers between min: 1 and max: 0
            return cls(1, 0)
        return cls(min(nums), max(nums))

    def __contains__(self, value: jvms.StackValue) -> bool:
        # Checks if the value is in the interval
        return self.min <= value.value <= self.max

    @classmethod
    def top(cls) -> Self:
        return cls(INT_MIN, INT_MAX)

    def __or__(self, other: "Interval") -> "Interval":
        if self.is_bot():
            return other
        elif other.is_bot():
            return self
        return wrap_or_top(min(self.min, other.min), max(self.max, other.max))

    def __and__(self, other: "Interval") -> "Interval":
        maxVal = min(self.max, other.max)
        minVal = max(self.min, other.min)

        if maxVal < minVal:
            return Interval(1, 0)

        return Interval(minVal, maxVal)

    def __lt__(self, other: "Interval") -> bool:
        if self == other:
            return False
        elif self.is_bot():
            return True
        elif other.is_bot():
            return False
        return other.min <= self.min and self.max <= other.max

    def __gt__(self, other: "Interval") -> bool:
        if self == other:
            return False
        elif self.is_bot():
            return False
        elif other.is_bot():
            return True
        return self.min <= other.min and other.max <= self.max

    def arithmetic(
        self, other: "Interval", opr: jvm.BinaryOpr
    ) -> list["Interval | str"]:
        min1 = self.min
        max1 = self.max
        min2 = other.min
        max2 = other.max

        match opr:
            case jvm.BinaryOpr.Add:
                return [wrap_or_top(min1 + min2, max1 + max2)]

            case jvm.BinaryOpr.Sub:
                return [wrap_or_top(min1 - max2, max1 - min2)]

            case jvm.BinaryOpr.Mul:
                temp = [min1 * min2, min1 * max2, max1 * min2, max1 * max2]
                return [wrap_or_top(min(temp), max(temp))]

            case jvm.BinaryOpr.Div:
                result: list[Interval | str] = []

                # Fx. we dont want [-3, 5] either
                if min2 <= 0 <= max2:
                    result.append("divide by zero")

                # Divide without 0
                # Fx. [-3, 5] is split into [-3, -1] and [1, 5]
                lo = Interval(min2, min(max2, -1))
                hi = Interval(max(min2, 1), max2)

                for part in (lo, hi):
                    if part.min <= part.max:
                        res = []
                        for x in (min1, max1):
                            for y in (part.min, part.max):
                                div = abs(x) // abs(y)
                                div = div if (x >= 0) == (y >= 0) else -div
                                res.append(div)
                        result.append(wrap_or_top(min(res), max(res)))

                return result

            case jvm.BinaryOpr.Rem:
                result = []

                # Fx. we dont want [-3, 5] either
                if min2 <= 0 <= max2:
                    result.append("divide by zero")

                if not (min2 == 0 == max2):
                    m = max(abs(min2), abs(max2)) - 1
                    lo = -m if min1 < 0 else 0
                    hi = m if max1 > 0 else 0
                    result.append(wrap_or_top(lo, hi))

                return result

            case _:
                # print("look here", opr, file=sys.stderr)
                raise NotImplementedError(f"TODO: {opr}")

    def compare(self, other: "Interval", opr: jvm.CmpOpr) -> Iterable[bool]:
        result = set()
        if self.is_bot() or other.is_bot():
            return result
        min1 = self.min
        max1 = self.max
        min2 = other.min
        max2 = other.max

        match opr:
            case jvm.CmpOpr.Le:
                can_true = min1 <= max2
                can_false = max1 > min2
            case jvm.CmpOpr.Lt:
                can_true = min1 < max2
                can_false = max1 >= min2
            case jvm.CmpOpr.Ge:
                can_true = max1 >= min2
                can_false = min1 < max2
            case jvm.CmpOpr.Gt:
                can_true = max1 > min2
                can_false = min1 <= max2
            case jvm.CmpOpr.Eq:
                can_true = not (self & other).is_bot()
                can_false = not (min1 == min2 == max1 == max2)
            case jvm.CmpOpr.Ne:
                can_true = not (min1 == min2 == max1 == max2)
                can_false = not (self & other).is_bot()
            case _:
                raise NotImplementedError(f"TODO: {opr}")
        if can_true:
            result.add(True)
        if can_false:
            result.add(False)
        return result

    def refine(
        self, other: "Interval", opr: jvm.CmpOpr, outcome: bool
    ) -> tuple["Interval", "Interval"]:
        if self.is_bot() or other.is_bot():
            return Interval.bot(), Interval.bot()

        min1 = self.min
        max1 = self.max
        min2 = other.min
        max2 = other.max

        # if outcome is false turn around the operators, to be the exact opposite
        if not outcome:
            opr = {
                jvm.CmpOpr.Le: jvm.CmpOpr.Gt,
                jvm.CmpOpr.Lt: jvm.CmpOpr.Ge,
                jvm.CmpOpr.Ge: jvm.CmpOpr.Lt,
                jvm.CmpOpr.Gt: jvm.CmpOpr.Le,
                jvm.CmpOpr.Eq: jvm.CmpOpr.Ne,
                jvm.CmpOpr.Ne: jvm.CmpOpr.Eq,
            }[opr]

        # self represents x
        # other represents y
        match opr:
            case jvm.CmpOpr.Le:  # x <= y
                na = self & Interval(INT_MIN, max2)
                nb = other & Interval(min1, INT_MAX)
            case jvm.CmpOpr.Lt:  # x < y
                na = self & Interval(INT_MIN, max2 - 1)
                nb = other & Interval(min1 + 1, INT_MAX)
            case jvm.CmpOpr.Ge:  # x >= y
                na = self & Interval(min2, INT_MAX)
                nb = other & Interval(INT_MIN, max1)
            case jvm.CmpOpr.Gt:  # x > y
                na = self & Interval(min2 + 1, INT_MAX)
                nb = other & Interval(INT_MIN, max1 - 1)
            case jvm.CmpOpr.Eq:  # x == y
                na = self & other
                nb = self & other
            case jvm.CmpOpr.Ne:  # x != y
                na = self
                nb = other
            case _:
                raise NotImplementedError(f"TODO: {opr}")

        return na, nb

    def widen(self, other: "Interval", K: set) -> "Interval":
        if self.is_bot():
            return other
        if other.is_bot():
            return self

        min1 = self.min
        max1 = self.max
        min2 = other.min
        max2 = other.max

        # Upper bound for widening with k >= max2
        if max2 > max1:
            mink = set()
            for k in K:
                if k >= max2:
                    mink.add(k)
            new_max = min(mink, default=INT_MAX)
        else:
            new_max = max1

        # Lower bound for widening with k <= min2
        if min2 < min1:
            maxk = set()
            for k in K:
                if k <= min2:
                    maxk.add(k)
            new_min = max(maxk, default=INT_MIN)
        else:
            new_min = min1

        return Interval(new_min, new_max)

    def split(self, K: set) -> list:
        min = self.min
        max = self.max

        KSorted = sorted(K)

        cursor = min
        interval_list = list()
        for k in KSorted:
            if cursor < k:
                interval_list.append(Interval(cursor, k - 1))
            interval_list.append(Interval(k, k))
            cursor = k + 1

        if cursor <= max:
            interval_list.append(Interval(cursor, max))

        return interval_list

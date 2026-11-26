from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

from tools import Abstraction, Lattice

import jvm
import jvm.state as jvms

INT_MIN = -(2**31)
INT_MAX = (2**31) - 1


# If the result can leave the 32-bit range, go to top.
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

    @classmethod
    def nonneg(cls) -> Self:
        return cls(0, INT_MAX)

    @classmethod
    def const(cls, c: int) -> Self:
        return cls(c, c)

    def is_bot(self) -> bool:
        return self.min > self.max

    @property
    def single(self) -> bool:
        return self.min == self.max

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

    def __contains__(self, value: jvms.StackValue | int) -> bool:
        # Checks if the value is in the interval
        v = value.value if isinstance(value, jvms.StackValue) else value
        return self.min <= v <= self.max

    @classmethod
    def top(cls) -> Self:
        return cls(INT_MIN, INT_MAX)

    def __or__(self, other: "Interval") -> "Interval":
        if self.is_bot():
            return other
        elif other.is_bot():
            return self
        return Interval(min(self.min, other.min), max(self.max, other.max))

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

        if self.is_bot() or other.is_bot():
            return [Interval.bot()]

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
                    bound = max(abs(min1), abs(max1))
                    lo = max(-m if min1 < 0 else 0, -bound)
                    hi = min(m if max1 > 0 else 0, bound)
                    result.append(wrap_or_top(lo, hi))

                return result

            case _:
                # print("look here", opr, file=sys.stderr)
                raise NotImplementedError(f"TODO: {opr}")

    def bitwise(self, other: "Interval", opr: jvm.BitOpr) -> "Interval":
        if self.is_bot() or other.is_bot():
            return Interval.bot()

        min1, max1 = self.min, self.max
        min2, max2 = other.min, other.max

        match opr:
            case jvm.BitOpr.And:
                if min1 >= 0 and min2 >= 0:
                    return Interval(0, min(max1, max2))
                if other.single and min2 >= 0:
                    return Interval(0, min2)
                if self.single and min1 >= 0:
                    return Interval(0, min1)
                return Interval.top()

            case jvm.BitOpr.Or:
                if min1 >= 0 and min2 >= 0:
                    max_val = max(max1, max2)
                    upper = (1 << max_val.bit_length()) - 1 if max_val > 0 else 0
                    return Interval(max(min1, min2), min(INT_MAX, upper))
                return Interval.top()

            case jvm.BitOpr.Xor:
                if self.single and other.single:
                    val = min1 ^ min2
                    val = ((val + (1 << 31)) % (1 << 32)) - (1 << 31)
                    return Interval(val, val)
                if min1 >= 0 and min2 >= 0:
                    max_val = max(max1, max2)
                    upper = (1 << max_val.bit_length()) - 1 if max_val > 0 else 0
                    return Interval(0, min(INT_MAX, upper))
                return Interval.top()

            case jvm.BitOpr.Shl:
                if other.single:
                    s = min2 & 0x1F
                    if s == 0:
                        return self
                    return wrap_or_top(min1 << s, max1 << s)
                return Interval.top()

            case jvm.BitOpr.Shr:
                if other.single:
                    s = min2 & 0x1F
                    return Interval(min1 >> s, max1 >> s)
                return Interval.top()

            case jvm.BitOpr.Ushr:
                if other.single:
                    s = min2 & 0x1F
                    if s == 0:
                        return self
                    if min1 >= 0:
                        return Interval(min1 >> s, max1 >> s)
                    return Interval(0, INT_MAX >> (s - 1))
                return Interval.nonneg() if min1 >= 0 else Interval.top()

            case _:
                return Interval.top()

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
                na = Interval.remove_endpoint(self, other)
                nb = Interval.remove_endpoint(other, self)
            case _:
                raise NotImplementedError(f"TODO: {opr}")

        return na, nb

    @staticmethod
    def remove_endpoint(a: "Interval", b: "Interval") -> "Interval":
        if b.min != b.max:
            return a
        c = b.min
        if a.min == c:
            return Interval(a.min + 1, a.max)
        if a.max == c:
            return Interval(a.min, a.max - 1)
        return a

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

    def narrow(self, other: "Interval") -> "Interval":
        if self.is_bot() or other.is_bot():
            return Interval.bot()
        lo = other.min if self.min == INT_MIN else self.min
        hi = other.max if self.max == INT_MAX else self.max
        return Interval(lo, hi)

    def split(self, K: set) -> list:
        lo = self.min
        hi = self.max

        KSorted = sorted(k for k in K if lo <= k <= hi)

        cursor = lo
        interval_list = list()
        for k in KSorted:
            if cursor < k:
                interval_list.append(Interval(cursor, k - 1))
            interval_list.append(Interval(k, k))
            cursor = k + 1

        if cursor <= hi:
            interval_list.append(Interval(cursor, hi))

        return interval_list

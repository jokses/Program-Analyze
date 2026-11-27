from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

from absIntervals import INT_MAX, INT_MIN, Interval
from tools import Abstraction, Lattice

import jvm
import jvm.state as jvms

# Maximum outliers accepted
MAX_OUTLIERS = 7

# Maximum gap from an interval to be defined as an outlier
GAP = 8

FAR = INT_MAX - INT_MIN + 1


def gap(p: int, iv: Interval) -> int:
    if iv.is_bot():
        return FAR
    if p < iv.min:
        return iv.min - p
    if p > iv.max:
        return p - iv.max
    return 0


@dataclass(frozen=True)
class IntervalOutliers(Abstraction, Lattice):
    iv: Interval
    outliers: frozenset = frozenset()

    @classmethod
    def make(cls, pieces: Iterable[Interval]) -> "IntervalOutliers":

        # Sorting the intervals, such the smallest are the first
        piecesSorted = sorted(
            (p for p in pieces if not p.is_bot()), key=lambda p: p.min
        )

        # Merging the intervals that are close (defined by GAP) to eachother
        clusters: list[Interval] = []
        for p in piecesSorted:
            if clusters and p.min - clusters[-1].max <= GAP:
                clusters[-1] = clusters[-1] | p
            else:
                clusters.append(p)

        iv_merged = Interval.bot()
        outliersFound: set[int] = set()
        for c in clusters:
            if c.min == c.max:
                outliersFound.add(c.min)
            else:
                iv_merged = iv_merged | c

        while True:
            close = [o for o in outliersFound if gap(o, iv_merged) <= GAP]
            if not close:
                break
            for o in close:
                outliersFound.remove(o)
                iv_merged = iv_merged | Interval(o, o)

        while len(outliersFound) > MAX_OUTLIERS:
            o = min(outliersFound, key=lambda q: gap(q, iv_merged))
            outliersFound.remove(o)
            iv_merged = iv_merged | Interval(o, o)

        return cls(iv_merged, frozenset(outliersFound))

    @property
    def pieces(self) -> list[Interval]:
        out = [] if self.iv.is_bot() else [self.iv]
        return out + [Interval(p, p) for p in sorted(self.outliers)]

    @classmethod
    def bot(cls) -> Self:
        return cls(Interval.bot(), frozenset())

    @classmethod
    def top(cls) -> Self:
        return cls(Interval.top(), frozenset())

    @classmethod
    def abstract(cls, values: Iterable[jvms.StackValue]) -> Self:
        return cls.make(Interval(v.value, v.value) for v in values)

    # Does the given interval or set contain a given value
    def __contains__(self, value: jvms.StackValue) -> bool:
        v = value.value
        return self.iv.min <= v <= self.iv.max or v in self.outliers

    def __or__(self, other: "IntervalOutliers") -> "IntervalOutliers":
        return IntervalOutliers.make(self.pieces + other.pieces)

    def __and__(self, other: "IntervalOutliers") -> "IntervalOutliers":
        return IntervalOutliers.make(
            a & b for a in self.pieces for b in other.pieces
        )

    def hasValue(self, x: int) -> bool:
        in_iv = self.iv.min <= x <= self.iv.max
        in_outliers = x in self.outliers
        return in_iv or in_outliers

    def __le__(self, other: "IntervalOutliers") -> bool:
        # every outlier in self is in other
        for o in self.outliers:
            if not other.hasValue(o):
                return False

        if self.iv.is_bot():
            return True

        # the parts of self.iv that lie outside other.iv
        if other.iv.is_bot():
            leftovers = [(self.iv.min, self.iv.max)]
        else:
            leftovers = [
                (self.iv.min, min(self.iv.max, other.iv.min - 1)),
                (max(self.iv.min, other.iv.max + 1), self.iv.max),
            ]

        for lo, hi in leftovers:
            if lo > hi:
                continue
            if hi - lo + 1 > len(other.outliers):
                return False
            for i in range(lo, hi + 1):
                if not other.hasValue(i):
                    return False

        return True

    def __lt__(self, other: "IntervalOutliers") -> bool:
        return self <= other and not other <= self

    def __ge__(self, other: "IntervalOutliers") -> bool:
        return other <= self

    def __gt__(self, other: "IntervalOutliers") -> bool:
        return other < self

    def is_bot(self) -> bool:
        return self.iv.is_bot() and not self.outliers

    def __str__(self) -> str:
        parts = [] if self.iv.is_bot() else [str(self.iv)]
        if self.outliers:
            parts.append("{" + ", ".join(map(str, sorted(self.outliers))) + "}")
        return "∪".join(parts) or "⊥"

    # Uses the methods defined in the Interval class
    def arithmetic(
        self, other: "IntervalOutliers", opr: jvm.BinaryOpr
    ) -> list["IntervalOutliers | str"]:
        if self.is_bot() or other.is_bot():
            return [IntervalOutliers.bot()]

        results: list[Interval] = []
        errors: list[str] = []
        for a in self.pieces:
            for b in other.pieces:
                for r in a.arithmetic(b, opr):
                    if isinstance(r, str):
                        if r not in errors:
                            errors.append(r)
                    else:
                        results.append(r)

        out: list[IntervalOutliers | str] = list(errors)
        if results:
            out.append(IntervalOutliers.make(results))
        return out

    # Uses the methods defined in the Interval class
    def compare(self, other: "IntervalOutliers", opr: jvm.CmpOpr) -> Iterable[bool]:
        result: set[bool] = set()
        for a in self.pieces:
            for b in other.pieces:
                result |= set(a.compare(b, opr))
        return result

    # Uses the methods defined in the Interval class
    def refine(
        self, other: "IntervalOutliers", opr: jvm.CmpOpr, outcome: bool
    ) -> tuple["IntervalOutliers", "IntervalOutliers"]:
        na: list[Interval] = []
        nb: list[Interval] = []
        for a in self.pieces:
            for b in other.pieces:
                if outcome in a.compare(b, opr):
                    ra, rb = a.refine(b, opr, outcome)
                    na.append(ra)
                    nb.append(rb)
        return IntervalOutliers.make(na), IntervalOutliers.make(nb)

    def widen(self, other: "IntervalOutliers", K: set) -> "IntervalOutliers":
        if self.is_bot():
            return other
        if other.is_bot():
            return self

        newOutliers = other.iv
        for o in other.outliers - self.outliers:
            newOutliers = newOutliers | Interval(o, o)

        iv = self.iv.widen(newOutliers, K)
        outliers = frozenset(o for o in self.outliers if gap(o, iv) > 0)
        return IntervalOutliers(iv, outliers)

    def split(self, K: set) -> list[Interval]:
        out = self.iv.split(K) + [Interval(o, o) for o in self.outliers]
        return sorted(out, key=lambda i: i.min)

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import heapq
import json
import operator
import sys
import traceback
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Self

import absIntervals
from absOutliers import IntervalOutliers
from dynamic import comparisons_in_method, operands, range_of, split_points
from syntactic_constants import MethodConstants
from tools import Abstraction, Lattice

import jpamb
import jvm
import jvm.state as jvms
import sexpr
from jvm.state import PC

INT_MIN = -(2**31)
INT_MAX = (2**31) - 1
INT_TYPES = {"int", "i", "boolean", "z", "char", "c", "short", "s", "byte", "b"}
ALL_OUTCOMES = {"ok", "assertion error", "divide by zero", "null pointer", "out of bounds"}


class AnalysisError(Exception):
    pass


# The ISY constants of every case, written by
# python solutions/syntactic/src/syntactic_constants.py --all --out solutions/syntactic/constants.json
CONSTANTS_JSON = Path(__file__).resolve().parents[2] / "syntactic" / "constants.json"


@functools.cache
def syntactic_table() -> dict[str, dict]:
    return json.loads(CONSTANTS_JSON.read_text())


# The methods a case calls are not cases themselves and are not in the file.
def load_syntactic_constants(method_id: str) -> MethodConstants | None:
    entry = syntactic_table().get(method_id)
    if entry is None:
        return None
    return MethodConstants(frozenset(entry["constants"]), frozenset(entry["comparisons"]))


def wrap_or_top(lo: int, hi: int) -> "Interval":
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
    def bot(cls) -> Self:
        return cls(1, 0)

    @classmethod
    def top(cls) -> Self:
        return cls(INT_MIN, INT_MAX)

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

    def __str__(self) -> str:
        if self.is_bot():
            return "⊥"
        f = lambda v: "-inf" if v == INT_MIN else "+inf" if v == INT_MAX else str(v)
        return f"[{f(self.min)}, {f(self.max)}]"

    def __sexpr__(self) -> str:
        return str(self)

    @classmethod
    def abstract(cls, values: Iterable[jvms.StackValue]) -> Self:
        nums = [v.value for v in values]
        if not nums:
            return cls(1, 0)
        return cls(min(nums), max(nums))

    def __contains__(self, value: jvms.StackValue | int) -> bool:
        v = value.value if isinstance(value, jvms.StackValue) else value
        return self.min <= v <= self.max

    def __or__(self, other: "Interval") -> "Interval":
        if self.is_bot():
            return other
        if other.is_bot():
            return self
        return wrap_or_top(min(self.min, other.min), max(self.max, other.max))

    def __and__(self, other: "Interval") -> "Interval":
        max_val = min(self.max, other.max)
        min_val = max(self.min, other.min)
        if max_val < min_val:
            return Interval.bot()
        return Interval(min_val, max_val)

    def __lt__(self, other: "Interval") -> bool:
        if self == other:
            return False
        if self.is_bot():
            return True
        if other.is_bot():
            return False
        return other.min <= self.min and self.max <= other.max

    def __gt__(self, other: "Interval") -> bool:
        if self == other:
            return False
        if self.is_bot():
            return False
        if other.is_bot():
            return True
        return self.min <= other.min and other.max <= self.max

    def arithmetic(
        self, other: "Interval", opr: jvm.BinaryOpr
    ) -> list["Interval | str"]:
        if self.is_bot() or other.is_bot():
            return []

        min1, max1 = self.min, self.max
        min2, max2 = other.min, other.max

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
                if min2 <= 0 <= max2:
                    result.append("divide by zero")

                lo = Interval(min2, min(max2, -1))
                hi = Interval(max(min2, 1), max2)

                for part in (lo, hi):
                    if not part.is_bot():
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
                return [Interval.top()]

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
        result: set[bool] = set()
        if self.is_bot() or other.is_bot():
            return result
        min1, max1 = self.min, self.max
        min2, max2 = other.min, other.max

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

        min1, max1 = self.min, self.max
        min2, max2 = other.min, other.max

        if not outcome:
            opr = {
                jvm.CmpOpr.Le: jvm.CmpOpr.Gt,
                jvm.CmpOpr.Lt: jvm.CmpOpr.Ge,
                jvm.CmpOpr.Ge: jvm.CmpOpr.Lt,
                jvm.CmpOpr.Gt: jvm.CmpOpr.Le,
                jvm.CmpOpr.Eq: jvm.CmpOpr.Ne,
                jvm.CmpOpr.Ne: jvm.CmpOpr.Eq,
            }[opr]

        match opr:
            case jvm.CmpOpr.Le:
                na = self & Interval(INT_MIN, max2)
                nb = other & Interval(min1, INT_MAX)
            case jvm.CmpOpr.Lt:
                na = self & Interval(INT_MIN, max2 - 1)
                nb = other & Interval(min1 + 1, INT_MAX)
            case jvm.CmpOpr.Ge:
                na = self & Interval(min2, INT_MAX)
                nb = other & Interval(INT_MIN, max1)
            case jvm.CmpOpr.Gt:
                na = self & Interval(min2 + 1, INT_MAX)
                nb = other & Interval(INT_MIN, max1 - 1)
            case jvm.CmpOpr.Eq:
                na = self & other
                nb = self & other
            case jvm.CmpOpr.Ne:
                na, nb = self, other
                if other.single:
                    if min1 == min2:
                        na = Interval(min1 + 1, max1)
                    elif max1 == min2:
                        na = Interval(min1, max1 - 1)
                if self.single:
                    if min2 == min1:
                        nb = Interval(min2 + 1, max2)
                    elif max2 == min1:
                        nb = Interval(min2, max2 - 1)
            case _:
                raise NotImplementedError(f"TODO: {opr}")

        return na, nb

    def widen(
        self, other: "Interval", K: set[int] | frozenset[int] | list[int]
    ) -> "Interval":
        if self.is_bot():
            return other
        if other.is_bot():
            return self

        min1, max1 = self.min, self.max
        min2, max2 = other.min, other.max

        if max2 > max1:
            new_max = min((k for k in K if k >= max2), default=INT_MAX)
        else:
            new_max = max1

        if min2 < min1:
            new_min = max((k for k in K if k <= min2), default=INT_MIN)
        else:
            new_min = min1

        return Interval(new_min, new_max)

    def narrow(self, other: "Interval") -> "Interval":
        if self.is_bot() or other.is_bot():
            return Interval.bot()
        lo = other.min if self.min == INT_MIN else self.min
        hi = other.max if self.max == INT_MAX else self.max
        return Interval(lo, hi)

    def split(self, points: Iterable[int]) -> list["Interval"]:
        if self.is_bot():
            return []
        cursor = self.min
        interval_list: list[Interval] = []
        for k in sorted(set(points)):
            if k < self.min:
                continue
            if k > self.max:
                break
            if cursor < k:
                interval_list.append(Interval(cursor, k - 1))
            interval_list.append(Interval(k, k))
            cursor = k + 1

        if cursor <= self.max:
            interval_list.append(Interval(cursor, self.max))

        return interval_list


# A cast wraps a value that does not fit, like (short) 40000 == -25536. Only an
# interval that already fits keeps its bounds.
def cast_interval(iv: Interval, target: Interval) -> Interval:
    if iv.is_bot() or (target.min <= iv.min and iv.max <= target.max):
        return iv
    return target


STR_TO_CMPOPR = {
    "eq": jvm.CmpOpr.Eq,
    "ne": jvm.CmpOpr.Ne,
    "lt": jvm.CmpOpr.Lt,
    "ge": jvm.CmpOpr.Ge,
    "gt": jvm.CmpOpr.Gt,
    "le": jvm.CmpOpr.Le,
}


def backward_arith_left(op: jvm.BinaryOpr, target: Interval, right: Interval) -> Interval:
    if target.is_bot() or right.is_bot():
        return Interval.bot()
    match op:
        case jvm.BinaryOpr.Add:
            return wrap_or_top(target.min - right.max, target.max - right.min)
        case jvm.BinaryOpr.Sub:
            return wrap_or_top(target.min + right.min, target.max + right.max)
        case jvm.BinaryOpr.Mul:
            if right.single and right.min != 0:
                d = right.min
                lo = (target.min + d - 1) // d if target.min >= 0 else target.min // d
                hi = target.max // d if target.max >= 0 else (target.max - d + 1) // d
                return wrap_or_top(lo, hi) if lo <= hi else Interval.bot()
            return Interval.top()
        case jvm.BinaryOpr.Div:
            if right.single and right.min > 0:
                d = right.min
                lo = target.min * d if target.min >= 0 else (target.min - 1) * d + 1
                hi = (target.max + 1) * d - 1 if target.max >= 0 else target.max * d
                return wrap_or_top(lo, hi)
            return Interval.top()
        case _:
            return Interval.top()


# The outliers keep a value far from the interval as a single point, like 1234 next
# to [0, 4] in test_make_keeps_far_point_as_outlier, so a range that lies fully
# between them is not in the bound.
def outliers_from_interval(iv: Interval) -> IntervalOutliers:
    return IntervalOutliers.make([absIntervals.Interval(iv.min, iv.max)])


# A value is in both the interval and the outliers, so each one limits the other.
def reduce_pair(
    iv: Interval, out: IntervalOutliers
) -> tuple[Interval, IntervalOutliers]:
    out = IntervalOutliers.make(
        absIntervals.Interval(max(iv.min, p.min), min(iv.max, p.max))
        for p in out.pieces()
    )
    pieces = out.pieces()
    if not pieces:
        return Interval.bot(), out
    return Interval(min(p.min for p in pieces), max(p.max for p in pieces)), out


# The errors and the value of a op b on the outliers. The value is None when the
# outliers give only errors.
def outliers_arithmetic(
    a: IntervalOutliers, b: IntervalOutliers, op: jvm.BinaryOpr
) -> tuple[set[str], IntervalOutliers | None]:
    results = a.arithmetic(b, op)
    values = [r for r in results if not isinstance(r, str)]
    return {r for r in results if isinstance(r, str)}, next(iter(values), None)


# -INT_MIN does not fit, and Java gives INT_MIN back.
def negate_interval(iv: Interval) -> Interval:
    if iv.is_bot():
        return iv
    return wrap_or_top(-iv.max, -iv.min)


# Each piece goes through the same function as the interval. Trying only some values
# of a piece is not sound, like (byte) of [0, 300] reaching 127 in the middle.
def map_pieces(out: IntervalOutliers, fn) -> IntervalOutliers:
    pieces = (fn(Interval(piece.min, piece.max)) for piece in out.pieces())
    return IntervalOutliers.make(absIntervals.Interval(p.min, p.max) for p in pieces)


@dataclass(frozen=True)
class Val:
    iv: Interval
    src: Optional[int] = None
    may_null: bool = False
    may_nonnull: bool = True
    elem_iv: Interval = Interval.top()
    len_of: Optional[int] = None
    bounded_by: Optional[int] = None
    alloc_id: Optional[int] = None
    is_ref: bool = False
    # None gives the outliers of iv.
    out: IntervalOutliers | None = None

    def __post_init__(self):
        out = outliers_from_interval(self.iv) if self.out is None else self.out
        iv, out = reduce_pair(self.iv, out)
        object.__setattr__(self, "iv", iv)
        object.__setattr__(self, "out", out)

    def join(
        self,
        o: "Val",
        K: Optional[set[int] | list[int]] = None,
        narrow: bool = False,
    ) -> "Val":
        if narrow:
            iv = self.iv.narrow(o.iv)
            out = self.out
            elem_iv = self.elem_iv.narrow(o.elem_iv)
        elif K is not None:
            k_set = set(K)
            iv = self.iv.widen(o.iv, k_set)
            out = self.out.widen(o.out, k_set)
            elem_iv = self.elem_iv.widen(o.elem_iv, k_set)
        else:
            iv = self.iv | o.iv
            out = self.out | o.out
            elem_iv = self.elem_iv | o.elem_iv

        return Val(
            iv=iv,
            src=self.src if self.src == o.src else None,
            may_null=self.may_null or o.may_null,
            may_nonnull=self.may_nonnull or o.may_nonnull,
            elem_iv=elem_iv,
            len_of=self.len_of if self.len_of == o.len_of else None,
            bounded_by=self.bounded_by if self.bounded_by == o.bounded_by else None,
            alloc_id=self.alloc_id if self.alloc_id == o.alloc_id else None,
            is_ref=self.is_ref or o.is_ref,
            out=out,
        )

    def __str__(self) -> str:
        if self.may_null and not self.may_nonnull:
            return "null"
        prefix = "null|" if self.may_null else ""
        return f"{prefix}{self.iv}"


BOT_VAL = Val(Interval.bot(), None, False, False, Interval.bot(), None, None, None, False)
TOP_VAL = Val(Interval.top(), None, True, True, Interval.top(), None, None, None, False)


@dataclass(frozen=True)
class State(sexpr.AsSExpr):
    locals: tuple[Val, ...]
    stack: tuple[Val, ...]

    def __str__(self) -> str:
        return f"{', '.join(map(str, self.locals))}/{':'.join(map(str, self.stack))}"

    def join(
        self,
        other: "State",
        K: Optional[set[int] | list[int]] = None,
        narrow: bool = False,
    ) -> "State":
        if len(self.stack) != len(other.stack):
            raise AnalysisError(f"stack height mismatch: {len(self.stack)} vs {len(other.stack)}")
        n_loc = max(len(self.locals), len(other.locals))
        l1 = self.locals + (BOT_VAL,) * (n_loc - len(self.locals))
        l2 = other.locals + (BOT_VAL,) * (n_loc - len(other.locals))
        return State(
            tuple(a.join(b, K, narrow) for a, b in zip(l1, l2)),
            tuple(a.join(b, K, narrow) for a, b in zip(self.stack, other.stack)),
        )

    def push(self, v: Val) -> "State":
        return State(self.locals, self.stack + (v,))

    def pop(self, n: int = 1) -> tuple[tuple[Val, ...], "State"]:
        if n == 0:
            return (), self
        return self.stack[-n:], State(self.locals, self.stack[:-n])

    def load(self, idx: int) -> Val:
        return self.locals[idx] if idx < len(self.locals) else TOP_VAL

    def store(self, idx: int, v: Val, preserve_bound: bool = False) -> "State":
        locs = list(self.locals)
        if idx >= len(locs):
            locs.extend([BOT_VAL] * (idx - len(locs) + 1))
        locs[idx] = v
        new_locs = tuple(
            Val(
                x.iv,
                x.src,
                x.may_null,
                x.may_nonnull,
                x.elem_iv,
                None if x.len_of == idx else x.len_of,
                x.bounded_by if (preserve_bound or x.bounded_by != idx) else None,
                x.alloc_id,
                x.is_ref,
                x.out,
            )
            for x in locs
        )
        stk = tuple(
            Val(
                x.iv,
                None if x.src == idx else x.src,
                x.may_null,
                x.may_nonnull,
                x.elem_iv,
                None if x.len_of == idx else x.len_of,
                None if x.bounded_by == idx else x.bounded_by,
                x.alloc_id,
                x.is_ref,
                x.out,
            )
            for x in self.stack
        )
        return State(new_locs, stk)

    def forget_elements(self, refs: Iterable[Val]) -> "State":
        # A called method can write into the arrays it gets. After the call their
        # elements can be anything.
        arrays = [r for r in refs if r.is_ref]

        def forget(x: Val) -> Val:
            return replace(x, elem_iv=Interval.top())

        def passed(x: Val, index: int | None = None) -> bool:
            return any(
                (r.src is not None and r.src in (x.src, index))
                or (r.alloc_id is not None and r.alloc_id == x.alloc_id)
                for r in arrays
            )

        return State(
            tuple(forget(x) if passed(x, i) else x for i, x in enumerate(self.locals)),
            tuple(forget(x) if passed(x) else x for x in self.stack),
        )

    def update_array_elems(self, ref: Val, val_iv: Interval) -> "State":
        def _upd(x: Val) -> Val:
            matches_local = ref.src is not None and x.src == ref.src
            matches_alloc = ref.alloc_id is not None and x.alloc_id == ref.alloc_id
            if matches_local or matches_alloc:
                return Val(
                    x.iv,
                    x.src,
                    False,
                    True,
                    x.elem_iv | val_iv,
                    x.len_of,
                    x.bounded_by,
                    x.alloc_id,
                    True,
                    x.out,
                )
            return x

        locs = list(self.locals)
        if ref.src is not None and ref.src < len(locs):
            old_arr = locs[ref.src]
            locs[ref.src] = Val(
                old_arr.iv,
                old_arr.src,
                False,
                True,
                old_arr.elem_iv | val_iv,
                old_arr.len_of,
                old_arr.bounded_by,
                old_arr.alloc_id,
                True,
                old_arr.out,
            )
        new_locs = tuple(_upd(x) for x in locs)
        new_stk = tuple(_upd(x) for x in self.stack)
        return State(new_locs, new_stk)


class UnboundedIntervalAnalysis:
    def __init__(
        self,
        bc: jpamb.Bytecode,
        entry_method: jvm.AbsMethodID,
        inputs: jpamb.Input | None = None,
        param_ivs: dict[int, Interval] | None = None,
    ):
        self.bc = bc
        self.entry_method = entry_method
        self.inputs = inputs
        # The binary search in extension.py narrows int parameters, like n in
        # [-2**31, -1], the lower half of all ints.
        self.param_ivs = param_ivs or {}
        self.K_cache: dict[jvm.AbsMethodID, set[int]] = {}
        self.headers_cache: dict[jvm.AbsMethodID, set[int]] = {}
        self.states: dict[PC, State] = {}
        self.outcomes: set[str] = set()
        self.preds: dict[PC, list[PC]] = {}

        self.return_vals: dict[jvm.AbsMethodID, Val] = {}
        self.call_sites: dict[jvm.AbsMethodID, set[PC]] = {}
        self.loop_edges: set[tuple[PC, PC]] = set()
        self.has_unknown_call: bool = False

    def _thresholds(self, m: jvm.AbsMethodID) -> set[int]:
        if m not in self.K_cache:
            ks = {INT_MIN, INT_MAX, -1, 0, 1}

            syn = load_syntactic_constants(m.encode())
            if syn is not None:
                ks |= set(syn.with_neighbours())
                ks |= set(syn.comparisons)

            method = self.bc.getmethod(m)
            for opr in method.opcodes:
                match opr:
                    case jvm.Push(value=v):
                        raw = getattr(v, "value", v)
                        if isinstance(raw, int) and not isinstance(raw, bool):
                            ks |= {raw - 1, raw, raw + 1}
                    case jvm.Incr(amount=a):
                        ks.add(a)

            self.K_cache[m] = {k for k in ks if INT_MIN <= k <= INT_MAX}
        return self.K_cache[m]

    def _loop_headers(self, m: jvm.AbsMethodID) -> set[int]:
        if m not in self.headers_cache:
            hs = set()
            method = self.bc.getmethod(m)
            for idx, opr in enumerate(method.opcodes):
                pc = PC(m, idx)
                t = getattr(opr, "target", None)
                if t is not None:
                    tgt_pc = pc % t
                    if tgt_pc.offset <= pc.offset:
                        hs.add(tgt_pc.offset)
            self.headers_cache[m] = hs
        return self.headers_cache[m]

    def _initial_state(self) -> State:
        method = self.bc.getmethod(self.entry_method)
        locs = [BOT_VAL] * method.max_locals
        params = self.entry_method.extension.params

        if self.inputs is None:
            for i, p in enumerate(params):
                match p:
                    case jvm.Int():
                        locs[i] = Val(Interval.top())
                    case jvm.Short():
                        locs[i] = Val(Interval(-32768, 32767))
                    case jvm.Byte():
                        locs[i] = Val(Interval(-128, 127))
                    case jvm.Char():
                        locs[i] = Val(Interval(0, 65535))
                    case jvm.Boolean():
                        locs[i] = Val(Interval(0, 1))
                    case jvm.Array(contains=jvm.Char()):
                        locs[i] = Val(
                            Interval.nonneg(),
                            may_null=True,
                            may_nonnull=True,
                            elem_iv=Interval(0, 65535),
                            is_ref=True,
                        )
                    case jvm.Array():
                        locs[i] = Val(
                            Interval.nonneg(),
                            may_null=True,
                            may_nonnull=True,
                            elem_iv=Interval.top(),
                            is_ref=True,
                        )
                    case jvm.Object(classname) if classname.dotted() == "java.lang.String":
                        locs[i] = Val(
                            Interval.nonneg(),
                            may_null=True,
                            may_nonnull=True,
                            elem_iv=Interval(0, 65535),
                            is_ref=True,
                        )
                    case _:
                        p_str = str(p)
                        if p_str in ("I", "int"):
                            locs[i] = Val(Interval.top())
                        elif p_str in ("Z", "boolean"):
                            locs[i] = Val(Interval(0, 1))
                        elif p_str.startswith("[C"):
                            locs[i] = Val(
                                Interval.nonneg(),
                                may_null=True,
                                may_nonnull=True,
                                elem_iv=Interval(0, 65535),
                                is_ref=True,
                            )
                        elif p_str.startswith("["):
                            locs[i] = Val(
                                Interval.nonneg(),
                                may_null=True,
                                may_nonnull=True,
                                elem_iv=Interval.top(),
                                is_ref=True,
                            )
                        else:
                            locs[i] = Val(
                                Interval.nonneg(),
                                may_null=True,
                                may_nonnull=True,
                                elem_iv=Interval.top(),
                                is_ref=True,
                            )
        else:
            for i, x in enumerate(self.inputs.values):
                match x:
                    case jpamb.case.Boolean(value=v):
                        locs[i] = Val(Interval.const(1 if v else 0))
                    case jpamb.case.Int(value=v):
                        locs[i] = Val(Interval.const(int(v)))
                    case jpamb.case.Array(contains=jvm.Char(), values=vals):
                        ords = [ord(a) for a in vals]
                        e_iv = Interval(min(ords), max(ords)) if ords else Interval.bot()
                        locs[i] = Val(
                            Interval.const(len(vals)),
                            may_null=False,
                            may_nonnull=True,
                            elem_iv=e_iv,
                            is_ref=True,
                        )
                    case jpamb.case.Array(values=vals):
                        e_iv = (
                            Interval(min(int(v) for v in vals), max(int(v) for v in vals))
                            if vals
                            else Interval.bot()
                        )
                        locs[i] = Val(
                            Interval.const(len(vals)),
                            may_null=False,
                            may_nonnull=True,
                            elem_iv=e_iv,
                            is_ref=True,
                        )
                    case jpamb.case.String(value=s):
                        ords = [ord(c) for c in s]
                        e_iv = Interval(min(ords), max(ords)) if ords else Interval.bot()
                        locs[i] = Val(
                            Interval.const(len(s)),
                            may_null=False,
                            may_nonnull=True,
                            elem_iv=e_iv,
                            is_ref=True,
                        )
                    case _:
                        locs[i] = Val(Interval.top(), may_null=True, may_nonnull=True, is_ref=True)

        for i, iv in self.param_ivs.items():
            locs[i] = replace(locs[i], iv=locs[i].iv & iv)
        return State(tuple(locs), ())

    def run(self) -> "UnboundedIntervalAnalysis":
        entry_pc = PC(self.entry_method, 0)
        init_st = self._initial_state()
        self.states = {entry_pc: init_st}
        work: list[tuple[str, int, PC]] = [(str(entry_pc.method), entry_pc.offset, entry_pc)]
        queued: set[PC] = {entry_pc}

        # The fixpoint is widened with the constants K at loop heads.
        while work:
            _, _, pc = heapq.heappop(work)
            queued.discard(pc)
            st = self.states[pc]

            for res in self.step(pc, st):
                if isinstance(res, str):
                    self.outcomes.add(res)
                    continue

                succ_pc, succ_st = res
                self.preds.setdefault(succ_pc, []).append(pc)

                if succ_pc.method == pc.method and succ_pc.offset <= pc.offset:
                    self.loop_edges.add((pc, succ_pc))

                old = self.states.get(succ_pc)
                if old is None:
                    self.states[succ_pc] = succ_st
                else:
                    headers = self._loop_headers(succ_pc.method)
                    K = (
                        self._thresholds(succ_pc.method)
                        if succ_pc.offset in headers
                        else None
                    )
                    new = old.join(succ_st, K, narrow=False)
                    if new == old:
                        continue
                    self.states[succ_pc] = new

                if succ_pc not in queued:
                    queued.add(succ_pc)
                    heapq.heappush(work, (str(succ_pc.method), succ_pc.offset, succ_pc))

        # Narrowing then takes back some of what widening added.
        for _ in range(2):
            incoming: dict[PC, State] = {entry_pc: init_st}
            for pc in sorted(self.states.keys(), key=lambda p: (str(p.method), p.offset)):
                st = self.states[pc]
                for res in self.step(pc, st):
                    if isinstance(res, str):
                        self.outcomes.add(res)
                        continue
                    succ_pc, succ_st = res
                    if succ_pc in incoming:
                        incoming[succ_pc] = incoming[succ_pc].join(succ_st, narrow=False)
                    else:
                        incoming[succ_pc] = succ_st

            changed = False
            for pc, inc_st in incoming.items():
                if pc in self.states:
                    narrowed = self.states[pc].join(inc_st, narrow=True)
                    if narrowed != self.states[pc]:
                        self.states[pc] = narrowed
                        changed = True
            if not changed:
                break

        return self

    def may_diverge(self) -> bool:
        if self.has_unknown_call:
            return True
        if not self.loop_edges:
            return False

        for tail_pc, header_pc in self.loop_edges:
            m = header_pc.method
            loop_pcs = [
                PC(m, off)
                for off in range(header_pc.offset, tail_pc.offset + 1)
                if PC(m, off) in self.states
            ]
            has_reachable_exit = False
            for lpc in loop_pcs:
                for res in self.step(lpc, self.states[lpc]):
                    if isinstance(res, str):
                        has_reachable_exit = True
                    else:
                        tgt, _ = res
                        if tgt.method == m and (
                            tgt.offset < header_pc.offset or tgt.offset > tail_pc.offset
                        ):
                            has_reachable_exit = True

            if not has_reachable_exit:
                return True

            for lpc in loop_pcs:
                opr = self.bc[lpc]
                if isinstance(opr, (jvm.If, jvm.Ifz)):
                    st_at_branch = self.states.get(lpc)
                    if st_at_branch and st_at_branch.stack:
                        top_iv = st_at_branch.stack[-1].iv
                        if top_iv.min == INT_MIN or top_iv.max == INT_MAX:
                            return True

        return False

    def _wake_callers(self, callee: jvm.AbsMethodID, ret_val: Val) -> list[tuple[PC, State]]:
        old_ret = self.return_vals.get(callee, BOT_VAL)
        joined_ret = old_ret.join(ret_val)
        if joined_ret == old_ret:
            return []
        self.return_vals[callee] = joined_ret
        woken: list[tuple[PC, State]] = []
        for call_pc in self.call_sites.get(callee, set()):
            caller_st = self.states.get(call_pc)
            if caller_st is None:
                continue
            opr = self.bc[call_pc]
            nargs = len(callee.extension.params) + (0 if isinstance(opr, jvm.InvokeStatic) else 1)
            args, after = caller_st.pop(nargs)
            woken.append((call_pc + 1, after.forget_elements(args).push(joined_ret)))
        return woken

    def step(self, pc: PC, state: State) -> list[tuple[PC, State] | str]:
        opr = self.bc[pc]
        out: list[tuple[PC, State] | str] = []

        match opr:
            case jvm.Push(type=t, value=v):
                raw = getattr(v, "value", v)
                match t:
                    case jvm.Int():
                        va = Val(Interval.const(int(raw)))
                    case jvm.Reference() | jvm.Array():
                        va = Val(Interval.bot(), may_null=True, may_nonnull=False, is_ref=True)
                    case jvm.Object(classname) if classname.dotted() == "java.lang.String":
                        s = str(raw)
                        ords = [ord(c) for c in s]
                        e_iv = Interval(min(ords), max(ords)) if ords else Interval.bot()
                        va = Val(
                            Interval.const(len(s)),
                            may_null=False,
                            may_nonnull=True,
                            elem_iv=e_iv,
                            is_ref=True,
                        )
                    case _:
                        if raw is None:
                            va = Val(Interval.bot(), may_null=True, may_nonnull=False, is_ref=True)
                        elif isinstance(raw, (bool, int)):
                            va = Val(Interval.const(int(raw)))
                        else:
                            va = Val(Interval.top(), may_null=False, may_nonnull=True, is_ref=True)
                out.append((pc + 1, state.push(va)))

            case jvm.Load(index=i):
                v = state.load(i)
                out.append(
                    (
                        pc + 1,
                        state.push(
                            Val(
                                v.iv,
                                i,
                                v.may_null,
                                v.may_nonnull,
                                v.elem_iv,
                                v.len_of,
                                v.bounded_by,
                                v.alloc_id,
                                v.is_ref,
                                v.out,
                            )
                        ),
                    )
                )

            case jvm.Store(index=i):
                [val], after = state.pop(1)
                stored = Val(
                    val.iv,
                    None,
                    val.may_null,
                    val.may_nonnull,
                    val.elem_iv,
                    val.len_of,
                    val.bounded_by,
                    val.alloc_id,
                    val.is_ref,
                    val.out,
                )
                out.append((pc + 1, after.store(i, stored)))

            case jvm.Incr(index=i, amount=a):
                v = state.load(i)
                _, inc_out = outliers_arithmetic(
                    v.out, outliers_from_interval(Interval.const(a)), jvm.BinaryOpr.Add
                )
                keep_bound = v.bounded_by if a <= 0 else None
                for r in v.iv.arithmetic(Interval.const(a), jvm.BinaryOpr.Add):
                    if isinstance(r, Interval):
                        new = Val(r, bounded_by=keep_bound, out=inc_out)
                        if not new.iv.is_bot():
                            out.append((pc + 1, state.store(i, new)))

            case jvm.Dup(words=w):
                if w == 1:
                    [val], _ = state.pop(1)
                    out.append((pc + 1, state.push(val)))
                elif w == 2:
                    [v2, v1], _ = state.pop(2)
                    out.append((pc + 1, state.push(v2).push(v1)))

            case jvm.Pop(words=w):
                _, after = state.pop(w)
                out.append((pc + 1, after))

            case jvm.Negate():
                [val], after = state.pop(1)
                iv = negate_interval(val.iv)
                neg_out = map_pieces(val.out, negate_interval)
                out.append((pc + 1, after.push(Val(iv, out=neg_out))))

            case jvm.Cast(to_=to):
                [val], after = state.pop(1)
                match to:
                    case jvm.Short():
                        target = Interval(-32768, 32767)
                    case jvm.Byte():
                        target = Interval(-128, 127)
                    case jvm.Char():
                        target = Interval(0, 65535)
                    case _:
                        target = Interval.top()
                iv = cast_interval(val.iv, target)
                cast_out = map_pieces(val.out, lambda piece: cast_interval(piece, target))
                out.append((pc + 1, after.push(Val(iv, out=cast_out))))

            case jvm.Binary(operant=op):
                [a, b], after = state.pop(2)
                results = a.iv.arithmetic(b.iv, op)
                out_errors, value_out = outliers_arithmetic(a.out, b.out, op)
                # An error needs both the interval and the outliers to allow it.
                out.extend(sorted({r for r in results if isinstance(r, str)} & out_errors))
                values = [r for r in results if isinstance(r, Interval) and not r.is_bot()]
                if values:
                    value = Val(functools.reduce(operator.or_, values), out=value_out)
                    if not value.iv.is_bot():
                        out.append((pc + 1, after.push(value)))

            case jvm.Bitopr(operant=op):
                [a, b], after = state.pop(2)
                res_iv = a.iv.bitwise(b.iv, op)
                if not res_iv.is_bot():
                    out.append((pc + 1, after.push(Val(res_iv))))

            case jvm.Get(static=True, field=field):
                if getattr(field.extension, "name", "") == "$assertionsDisabled":
                    out.append((pc + 1, state.push(Val(Interval.const(0)))))
                else:
                    out.append((pc + 1, state.push(TOP_VAL)))

            case jvm.New(classname=cls):
                if "AssertionError" in str(cls):
                    out.append("assertion error")
                out.append(
                    (
                        pc + 1,
                        state.push(
                            Val(Interval.top(), may_null=False, may_nonnull=True, is_ref=True)
                        ),
                    )
                )

            case jvm.NewArray(dim=d):
                sizes, after = state.pop(d)
                if sizes[0].iv.min < 0:
                    out.append("negative array size")
                    out.append("out of bounds")
                length_iv = sizes[0].iv & Interval.nonneg()
                if not length_iv.is_bot():
                    init_elem = Interval.const(0) if length_iv.max > 0 else Interval.bot()
                    out.append(
                        (
                            pc + 1,
                            after.push(
                                Val(
                                    length_iv,
                                    may_null=False,
                                    may_nonnull=True,
                                    elem_iv=init_elem,
                                    alloc_id=pc.offset,
                                    is_ref=True,
                                )
                            ),
                        )
                    )

            case jvm.ArrayLength():
                [ref], after = state.pop(1)
                if ref.may_null:
                    out.append("null pointer")
                if ref.may_nonnull:
                    len_iv = (
                        (ref.iv & Interval.nonneg())
                        if not ref.iv.is_bot()
                        else Interval.nonneg()
                    )
                    out.append((pc + 1, after.push(Val(len_iv, len_of=ref.src))))

            case jvm.ArrayLoad():
                [ref, idx], after = state.pop(2)
                if ref.may_null:
                    out.append("null pointer")
                if ref.may_nonnull:
                    arr_len = (
                        (ref.iv & Interval.nonneg())
                        if not ref.iv.is_bot()
                        else Interval.nonneg()
                    )
                    is_provably_in_bounds = (
                        ref.src is not None
                        and idx.bounded_by == ref.src
                        and idx.iv.min >= 0
                    )
                    if not is_provably_in_bounds:
                        if idx.iv.min < 0 or idx.iv.max >= arr_len.min:
                            out.append("out of bounds")
                    valid_idx = (
                        (idx.iv & Interval(0, arr_len.max - 1))
                        if arr_len.max > 0
                        else Interval.bot()
                    )
                    if not valid_idx.is_bot():
                        loaded_iv = (
                            ref.elem_iv if not ref.elem_iv.is_bot() else Interval.top()
                        )
                        out.append((pc + 1, after.push(Val(loaded_iv))))

            case jvm.ArrayStore():
                [ref, idx, val], after = state.pop(3)
                if ref.may_null:
                    out.append("null pointer")
                if ref.may_nonnull:
                    arr_len = (
                        (ref.iv & Interval.nonneg())
                        if not ref.iv.is_bot()
                        else Interval.nonneg()
                    )
                    is_provably_in_bounds = (
                        ref.src is not None
                        and idx.bounded_by == ref.src
                        and idx.iv.min >= 0
                    )
                    if not is_provably_in_bounds:
                        if idx.iv.min < 0 or idx.iv.max >= arr_len.min:
                            out.append("out of bounds")
                    valid_idx = (
                        (idx.iv & Interval(0, arr_len.max - 1))
                        if arr_len.max > 0
                        else Interval.bot()
                    )
                    if not valid_idx.is_bot():
                        next_st = after.update_array_elems(ref, val.iv)
                        out.append((pc + 1, next_st))

            case jvm.CheckCast():
                [ref], after = state.pop(1)
                out.append((pc + 1, after.push(ref)))

            case jvm.Goto(target=t):
                out.append((pc % t, state))

            case jvm.Return(type=ret_type):
                out.append("ok")
                if ret_type is not None and state.stack:
                    out.extend(self._wake_callers(pc.method, state.stack[-1]))

            case jvm.Throw():
                out.append("assertion error")

            case jvm.Ifz(condition=cond, target=target):
                [a], after = state.pop(1)
                out.extend(self._branch(pc, after, a, Val(Interval.const(0)), cond, target))

            case jvm.If(condition=cond, target=target):
                [a, b], after = state.pop(2)
                out.extend(self._branch(pc, after, a, b, cond, target))

            case jvm.InvokeDynamic(method=m):
                nargs = len(m.extension.params)
                _, after = state.pop(nargs)
                returns_void = m.extension.return_type is None or str(m).endswith("V")
                ret_val = Val(
                    Interval.nonneg(),
                    may_null=False,
                    may_nonnull=True,
                    elem_iv=Interval(0, 65535),
                    is_ref=True,
                )
                out.append((pc + 1, after if returns_void else after.push(ret_val)))

            case jvm.InvokeStatic(method=m):
                nargs = len(m.extension.params)
                args, after = state.pop(nargs)
                after = after.forget_elements(args)
                returns_void = m.extension.return_type is None or str(m).endswith("V")
                try:
                    callee = self.bc.getmethod(m)
                    if not returns_void:
                        self.call_sites.setdefault(m, set()).add(pc)
                        if m in self.return_vals:
                            out.append((pc + 1, after.push(self.return_vals[m])))
                    else:
                        out.append((pc + 1, after))
                    locs = list(args) + [BOT_VAL] * (callee.max_locals - nargs)
                    out.append((PC(m, 0), State(tuple(locs), ())))
                except Exception:
                    self.has_unknown_call = True
                    self.outcomes.update(ALL_OUTCOMES)
                    out.append((pc + 1, after if returns_void else after.push(TOP_VAL)))

            case (
                jvm.InvokeVirtual(method=m)
                | jvm.InvokeSpecial(method=m)
                | jvm.InvokeInterface(method=m)
            ):
                nargs = len(m.extension.params) + 1
                args, after = state.pop(nargs)
                # A String cannot change, but getChars writes into a char[].
                if m.classname.dotted() == "java.lang.String":
                    after = after.forget_elements(args[1:])
                else:
                    after = after.forget_elements(args)
                recv_arg = args[0]
                if recv_arg.may_null:
                    out.append("null pointer")
                if recv_arg.may_nonnull:
                    returns_void = m.extension.return_type is None or str(m).endswith("V")
                    classname_str = m.classname.dotted()
                    method_name = m.extension.name

                    if classname_str == "java.lang.String":
                        match method_name:
                            case "equals":
                                out.append((pc + 1, after.push(Val(Interval(0, 1)))))
                            case "length":
                                str_len = (
                                    (recv_arg.iv & Interval.nonneg())
                                    if not recv_arg.iv.is_bot()
                                    else Interval.nonneg()
                                )
                                out.append(
                                    (pc + 1, after.push(Val(str_len, len_of=recv_arg.src)))
                                )
                            case "charAt":
                                idx_arg = args[1]
                                str_len = (
                                    (recv_arg.iv & Interval.nonneg())
                                    if not recv_arg.iv.is_bot()
                                    else Interval.nonneg()
                                )
                                if idx_arg.iv.min < 0 or idx_arg.iv.max >= str_len.min:
                                    out.append("out of bounds")
                                ch_iv = (
                                    recv_arg.elem_iv
                                    if not recv_arg.elem_iv.is_bot()
                                    else Interval(0, 65535)
                                )
                                out.append((pc + 1, after.push(Val(ch_iv))))
                            case _:
                                out.append(
                                    (pc + 1, after if returns_void else after.push(TOP_VAL))
                                )
                    elif classname_str == "jpamb.utils.Smith":
                        self.has_unknown_call = True
                        if method_name in ("nextBoolean", "flip"):
                            out.append((pc + 1, after.push(Val(Interval(0, 1)))))
                        else:
                            out.append(
                                (
                                    pc + 1,
                                    after if returns_void else after.push(Val(Interval.top())),
                                )
                            )
                    else:
                        try:
                            callee = self.bc.getmethod(m)
                            if not returns_void:
                                self.call_sites.setdefault(m, set()).add(pc)
                                if m in self.return_vals:
                                    out.append((pc + 1, after.push(self.return_vals[m])))
                            else:
                                out.append((pc + 1, after))
                            recv = Val(
                                recv_arg.iv,
                                0,
                                False,
                                True,
                                recv_arg.elem_iv,
                                is_ref=True,
                                out=recv_arg.out,
                            )
                            locs = (
                                [recv]
                                + list(args[1:])
                                + [BOT_VAL] * (callee.max_locals - nargs)
                            )
                            out.append((PC(m, 0), State(tuple(locs), ())))
                        except Exception:
                            m_str = str(m)
                            if (
                                "Object.<init>" in m_str
                                or "AssertionError.<init>" in m_str
                                or "StringBuilder." in m_str
                            ):
                                out.append(
                                    (pc + 1, after if returns_void else after.push(TOP_VAL))
                                )
                            else:
                                self.has_unknown_call = True
                                self.outcomes.update(ALL_OUTCOMES)
                                out.append(
                                    (pc + 1, after if returns_void else after.push(TOP_VAL))
                                )

            case _:
                self.has_unknown_call = True
                self.outcomes.update(ALL_OUTCOMES)

        return out

    def _is_ref_test(self, pc: PC, a: Val) -> bool:
        if a.may_null or a.is_ref:
            return True
        if pc.offset > 0:
            prev = self.bc[PC(pc.method, pc.offset - 1)]
            if isinstance(prev, jvm.Load):
                return str(prev.type).lower() not in INT_TYPES
        return False

    def _branch(
        self, pc: PC, after: State, a: Val, b: Val, cond, target: int
    ) -> list[tuple[PC, State]]:
        cond_str = str(cond)
        opr = self.bc[pc]

        # ifnull and ifnonnull have the conditions is and isnot.
        reference_condition = cond_str in ("is", "isnot")
        null_test = reference_condition or (
            cond_str in ("eq", "ne") and self._is_ref_test(pc, a)
        )
        if isinstance(opr, jvm.Ifz) and null_test:
            is_null_branch = cond_str in ("eq", "is")
            res: list[tuple[PC, State]] = []
            for taken in (True, False):
                tgt_pc = (pc % target) if taken else (pc + 1)
                expect_null = is_null_branch if taken else (not is_null_branch)
                if expect_null:
                    if not a.may_null:
                        continue
                    s = (
                        after.store(
                            a.src, Val(Interval.bot(), a.src, True, False, is_ref=True)
                        )
                        if a.src is not None
                        else after
                    )
                else:
                    if not a.may_nonnull:
                        continue
                    s = (
                        after.store(
                            a.src,
                            Val(
                                a.iv,
                                a.src,
                                False,
                                True,
                                a.elem_iv,
                                a.len_of,
                                a.bounded_by,
                                a.alloc_id,
                                True,
                                a.out,
                            ),
                        )
                        if a.src is not None
                        else after
                    )
                res.append((tgt_pc, s))
            return res

        # if_acmpeq and if_acmpne compare references and give no int range.
        unknown = a.may_null or b.may_null or a.is_ref or b.is_ref
        if isinstance(opr, jvm.If) and (reference_condition or unknown):
            return [(pc % target, after), (pc + 1, after)]

        cmp_opr = cond if isinstance(cond, jvm.CmpOpr) else STR_TO_CMPOPR[cond_str]
        outcomes = set(a.iv.compare(b.iv, cmp_opr)) & set(a.out.compare(b.out, cmp_opr))

        res = []
        for taken in (True, False):
            if taken not in outcomes:
                continue
            tgt_pc = (pc % target) if taken else (pc + 1)
            ra_iv, rb_iv = a.iv.refine(b.iv, cmp_opr, taken)
            ra_out, rb_out = a.out.refine(b.out, cmp_opr, taken)
            ra_iv, ra_out = reduce_pair(ra_iv, ra_out)
            rb_iv, rb_out = reduce_pair(rb_iv, rb_out)

            if ra_iv.is_bot() or rb_iv.is_bot():
                continue

            eff_opr = (
                cmp_opr
                if taken
                else {
                    jvm.CmpOpr.Le: jvm.CmpOpr.Gt,
                    jvm.CmpOpr.Lt: jvm.CmpOpr.Ge,
                    jvm.CmpOpr.Ge: jvm.CmpOpr.Lt,
                    jvm.CmpOpr.Gt: jvm.CmpOpr.Le,
                    jvm.CmpOpr.Eq: jvm.CmpOpr.Ne,
                    jvm.CmpOpr.Ne: jvm.CmpOpr.Eq,
                }[cmp_opr]
            )

            s = after
            a_bound = (
                b.len_of
                if (eff_opr == jvm.CmpOpr.Lt and b.len_of is not None)
                else a.bounded_by
            )
            b_bound = (
                a.len_of
                if (eff_opr == jvm.CmpOpr.Gt and a.len_of is not None)
                else b.bounded_by
            )

            if a.src is not None:
                s = s.store(
                    a.src,
                    Val(
                        ra_iv,
                        a.src,
                        a.may_null,
                        a.may_nonnull,
                        a.elem_iv,
                        a.len_of,
                        a_bound,
                        a.alloc_id,
                        a.is_ref,
                        ra_out,
                    ),
                    preserve_bound=True,
                )
            if b.src is not None:
                s = s.store(
                    b.src,
                    Val(
                        rb_iv,
                        b.src,
                        b.may_null,
                        b.may_nonnull,
                        b.elem_iv,
                        b.len_of,
                        b_bound,
                        b.alloc_id,
                        b.is_ref,
                        rb_out,
                    ),
                    preserve_bound=True,
                )
            res.append((tgt_pc, s))
        return res

    def reachable_range_tuples(
        self, split: str = "three", m: jvm.AbsMethodID | None = None
    ) -> frozenset[tuple[PC, int, int, int]]:
        target = m or self.entry_method
        points_cache: dict[jvm.AbsMethodID, list[int]] = {}
        tuples: set[tuple[PC, int, int, int]] = set()

        for pc, st in sorted(
            self.states.items(), key=lambda kv: (str(kv[0].method), kv[0].offset)
        ):
            if m is not None and pc.method != target:
                continue
            opr = self.bc[pc]
            ops = operands(opr, st.stack)
            if not ops:
                continue

            if pc.method not in points_cache:
                points_cache[pc.method] = split_points(
                    set(comparisons_in_method(pc.method)), split
                )
            pts = points_cache[pc.method]

            for position, val in enumerate(ops):
                if val.is_ref or (val.may_null and not val.may_nonnull) or val.iv.is_bot():
                    continue
                # With the points 0 and 300 the fuzzer records 7 as the range [1, 299].
                for piece in val.out.pieces():
                    for sub_iv in Interval(piece.min, piece.max).split(pts):
                        tuples.add((pc, position, *range_of(sub_iv.min, pts)))

        return frozenset(tuples)

    def solve_uncovered_range(
        self,
        target_tuple: tuple[PC, int, int, int],
        path: tuple[PC, ...],
        param_idx: int = 0,
    ) -> Interval:
        # We go backwards along a run that reached the target pc and undo a loop once
        # for every time that run went through it. In divisionLoop the run divides n by
        # 2 five times and n == 32 at the assertion gives n in [1024, 1055].
        target_pc, position, low, high = target_tuple
        st_at_target = self.states.get(target_pc)
        if target_pc not in path or st_at_target is None:
            return Interval.top()
        ops = operands(self.bc[target_pc], st_at_target.stack)
        # Only an operand loaded straight from the parameter can be followed back.
        if position >= len(ops) or ops[position].src != param_idx:
            return Interval.top()
        cur_iv = Interval(low, high)
        tracked_local = param_idx

        before = path[: path.index(target_pc)]
        steps = [pc for pc in before if pc.method == target_pc.method]
        k = len(steps) - 1
        while k >= 0:
            match self.bc[steps[k]]:
                case jvm.Store(index=i) if i == tracked_local:
                    # n = n op c is load n, push c, binary op and store n.
                    if k < 3:
                        return Interval.top()
                    load, push, binary = (self.bc[pc] for pc in steps[k - 3 : k])
                    if not (
                        isinstance(load, jvm.Load)
                        and load.index == tracked_local
                        and isinstance(push, jvm.Push)
                        and isinstance(push.type, jvm.Int)
                        and isinstance(binary, jvm.Binary)
                    ):
                        return Interval.top()
                    constant = Interval.const(push.value)
                    cur_iv = backward_arith_left(binary.operant, cur_iv, constant)
                    k -= 4
                    continue
                case jvm.Incr(index=i, amount=a) if i == tracked_local:
                    cur_iv = backward_arith_left(jvm.BinaryOpr.Add, cur_iv, Interval.const(a))
            k -= 1

        entry_st = self.states.get(PC(target_pc.method, 0))
        if entry_st and param_idx < len(entry_st.locals):
            cur_iv = cur_iv & entry_st.locals[param_idx].iv
        return cur_iv


@dataclass
class RangeCoverageOracle:
    analysis: UnboundedIntervalAnalysis
    upper_bound: frozenset[tuple[PC, int, int, int]]

    @classmethod
    def build(
        cls, bc: jpamb.Bytecode, methodid: jvm.AbsMethodID, split: str = "three"
    ) -> "RangeCoverageOracle":
        analysis = UnboundedIntervalAnalysis(bc, methodid, None).run()
        return cls(analysis, analysis.reachable_range_tuples(split=split))


def interpret():
    methodid, input_val, steps = jpamb.getcase(
        "unbounded",
        "1.0",
        "kindleframe",
        ["static", "interval", "python"],
        for_science=True,
    )
    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(suite, eff, {})
    analysis = UnboundedIntervalAnalysis(bc, methodid, input_val).run()

    x = jpamb.emit_init(analysis.states)
    for pc, st in sorted(analysis.states.items(), key=lambda kv: kv[0].offset):
        if steps <= 0:
            break
        x = jpamb.emit_step(x, pc, analysis.states, depth=1)
        steps -= 1
    for outcome in sorted(analysis.outcomes):
        if steps <= 0:
            break
        jpamb.emit_step(x, PC(methodid, 0), outcome, depth=1)
        steps -= 1


def analyse():
    methodid = jpamb.getmethodid(
        "unbounded",
        "1.0",
        "kindleframe",
        ["static", "interval", "python"],
        for_science=True,
    )
    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(suite, eff, {})

    try:
        analysis = UnboundedIntervalAnalysis(bc, methodid, None).run()
        final = analysis.outcomes
        diverge = analysis.may_diverge()
        incomplete = False
    except Exception as e:
        traceback.print_exc()
        print(f"analysis failed: {e!r}", file=sys.stderr)
        final = set()
        diverge = True
        incomplete = True

    for f in jpamb.QUERIES:
        may = incomplete or (f in final) or (f == "*" and diverge)
        print(f"{f};{'maybe' if may else 'no'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--interpret", action="store_true")
    args, _ = ap.parse_known_args()
    if args.interpret:
        interpret()
    else:
        analyse()

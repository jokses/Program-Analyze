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
from dataclasses import dataclass, field, replace
from pathlib import Path

from absIntervals import INT_MAX, INT_MIN, Interval, wrap_or_top
from absIntervalSets import IntervalSet
from absOutliers import IntervalOutliers
from absSigns import SignSet
from dynamic import comparisons_in_method, operands, range_of, split_points
from syntactic_constants import MethodConstants

import jpamb
import jvm
import sexpr
from jvm.state import PC

INT_TYPES = {"int", "i", "boolean", "z", "char", "c", "short", "s", "byte", "b"}
ALL_OUTCOMES = {
    "ok",
    "assertion error",
    "divide by zero",
    "null pointer",
    "out of bounds",
}
STRING = "java.lang.String"
# The values of Java's short, byte and char.
SHORT = Interval(-(2**15), 2**15 - 1)
BYTE = Interval(-(2**7), 2**7 - 1)
CHAR = Interval(0, 2**16 - 1)


class AnalysisError(Exception):
    pass


# The comparison that is true when the branch is not taken.
NEGATED = {
    jvm.CmpOpr.Le: jvm.CmpOpr.Gt,
    jvm.CmpOpr.Lt: jvm.CmpOpr.Ge,
    jvm.CmpOpr.Ge: jvm.CmpOpr.Lt,
    jvm.CmpOpr.Gt: jvm.CmpOpr.Le,
    jvm.CmpOpr.Eq: jvm.CmpOpr.Ne,
    jvm.CmpOpr.Ne: jvm.CmpOpr.Eq,
}


# The ISY constants of every case, written by
# python solutions/syntactic/src/syntactic_constants.py --all \
#     --out solutions/syntactic/constants.json
CONSTANTS_JSON = Path(__file__).resolve().parents[2] / "syntactic" / "constants.json"


@functools.cache
def syntactic_table() -> dict[str, dict]:
    return json.loads(CONSTANTS_JSON.read_text())


# The methods a case calls are not cases themselves and are not in the file.
def load_syntactic_constants(method_id: str) -> MethodConstants | None:
    entry = syntactic_table().get(method_id)
    if entry is None:
        return None
    return MethodConstants(
        frozenset(entry["constants"]), frozenset(entry["comparisons"])
    )


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


def backward_arith_left(
    op: jvm.BinaryOpr, target: Interval, right: Interval
) -> Interval:
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


# Next to its interval each value has pieces. A range between two pieces, like 0
# when n != 0, is not in the bound. --domain picks the pieces once for the whole run.
# IntervalOutliers cannot leave out a point inside an interval, and SignSet keeps
# only the signs, as the course's sign analysis does next to its intervals.
# The first domain is the default, as for --domain.
DOMAINS = {"sets": IntervalSet, "outliers": IntervalOutliers, "signs": SignSet}
DEFAULT_DOMAIN = next(iter(DOMAINS))
Pieces = DOMAINS[DEFAULT_DOMAIN]


def pieces_from_interval(iv: Interval) -> Pieces:
    return Pieces.make([iv])


# A value is in both the interval and the pieces, so each one limits the other.
def reduce_pair(iv: Interval, piece_set: Pieces) -> tuple[Interval, Pieces]:
    piece_set = type(piece_set).make(iv & piece for piece in piece_set.pieces)
    pieces = piece_set.pieces
    if not pieces:
        return Interval.bot(), piece_set
    low = min(piece.min for piece in pieces)
    high = max(piece.max for piece in pieces)
    return Interval(low, high) & iv, piece_set


# The errors and the value of left op right on the pieces. The value is None when the
# pieces give only errors.
def pieces_arithmetic(
    left: Pieces, right: Pieces, op: jvm.BinaryOpr
) -> tuple[set[str], Pieces | None]:
    results = left.arithmetic(right, op)
    errors = {result for result in results if isinstance(result, str)}
    values = [result for result in results if not isinstance(result, str)]
    return errors, next(iter(values), None)


# -INT_MIN does not fit and Java gives INT_MIN back, so such an interval becomes
# all ints.
def negate_interval(iv: Interval) -> Interval:
    if iv.is_bot():
        return iv
    return wrap_or_top(-iv.max, -iv.min)


# The smallest interval with every number in it, empty without numbers.
def interval_of(numbers) -> Interval:
    numbers = list(numbers)
    return Interval(min(numbers), max(numbers)) if numbers else Interval.bot()


# Methods by name, then their pcs in order.
def pc_order(pc: PC) -> tuple[str, int]:
    return (str(pc.method), pc.offset)


# Each piece goes through the same function as the interval. Trying only some values
# of a piece is not sound, like (byte) of [0, 300] reaching 127 in the middle.
def map_pieces(piece_set: Pieces, fn) -> Pieces:
    return type(piece_set).make(fn(piece) for piece in piece_set.pieces)


@dataclass(frozen=True)
class Val:
    iv: Interval
    src: int | None = None
    may_null: bool = False
    may_nonnull: bool = True
    elem_iv: Interval = field(default_factory=Interval.top)
    len_of: int | None = None
    bounded_by: int | None = None
    alloc_id: int | None = None
    is_ref: bool = False
    # None gives iv as one piece.
    piece_set: Pieces | None = None

    def __post_init__(self):
        piece_set = self.piece_set
        if piece_set is None:
            piece_set = pieces_from_interval(self.iv)
        iv, piece_set = reduce_pair(self.iv, piece_set)
        object.__setattr__(self, "iv", iv)
        object.__setattr__(self, "piece_set", piece_set)

    # Built when asked so the pieces come from the domain of the run.
    @classmethod
    def bot(cls) -> Val:
        return cls(Interval.bot(), may_nonnull=False, elem_iv=Interval.bot())

    @classmethod
    def top(cls) -> Val:
        return cls(Interval.top(), may_null=True)

    @classmethod
    def null(cls) -> Val:
        return cls(Interval.bot(), may_null=True, may_nonnull=False, is_ref=True)

    # A reference that is not null, to an array or a string of any length.
    @classmethod
    def reference(cls) -> Val:
        return cls(Interval.top(), is_ref=True)

    def join(
        self,
        other: Val,
        K: set[int] | list[int] | None = None,
        narrow: bool = False,
    ) -> Val:
        if narrow:
            iv = self.iv.narrow(other.iv)
            piece_set = self.piece_set
            elem_iv = self.elem_iv.narrow(other.elem_iv)
        elif K is not None:
            k_set = set(K)
            iv = self.iv.widen(other.iv, k_set)
            piece_set = self.piece_set.widen(other.piece_set, k_set)
            elem_iv = self.elem_iv.widen(other.elem_iv, k_set)
        else:
            iv = self.iv | other.iv
            piece_set = self.piece_set | other.piece_set
            elem_iv = self.elem_iv | other.elem_iv

        return Val(
            iv=iv,
            src=same_or_none(self.src, other.src),
            may_null=self.may_null or other.may_null,
            may_nonnull=self.may_nonnull or other.may_nonnull,
            elem_iv=elem_iv,
            len_of=same_or_none(self.len_of, other.len_of),
            bounded_by=same_or_none(self.bounded_by, other.bounded_by),
            alloc_id=same_or_none(self.alloc_id, other.alloc_id),
            is_ref=self.is_ref or other.is_ref,
            piece_set=piece_set,
        )

    def __str__(self) -> str:
        if self.may_null and not self.may_nonnull:
            return "null"
        prefix = "null|" if self.may_null else ""
        return f"{prefix}{self.iv}"


# A value two joined states agree on, or None when they differ.
def same_or_none(first: int | None, second: int | None) -> int | None:
    return first if first == second else None


# Two unknown (None) values do not count as equal.
def known_equal(first: int | None, second: int | None) -> bool:
    return first is not None and first == second


# Whether the instructions are load n, push an int and a binary op, as in
# n = n op c.
def is_update_by_constant(load, push, binary, local: int) -> bool:
    loads_local = isinstance(load, jvm.Load) and load.index == local
    pushes_int = isinstance(push, jvm.Push) and isinstance(push.type, jvm.Int)
    return loads_local and pushes_int and isinstance(binary, jvm.Binary)


# The length of an array or a string.
def length(ref: Val) -> Interval:
    return Interval.nonneg() if ref.iv.is_bot() else ref.iv & Interval.nonneg()


# Whether idx can be outside the array or string ref, and whether it can be inside.
def check_index(ref: Val, idx: Val) -> tuple[bool, bool]:
    array_length = length(ref)
    # bounded_by is the local whose length the index was compared with.
    proven_inside = known_equal(ref.src, idx.bounded_by) and idx.iv.min >= 0
    can_be_outside = idx.iv.min < 0 or idx.iv.max >= array_length.min
    indexes = Interval(0, array_length.max - 1)
    can_be_inside = not (idx.iv & indexes).is_bot()
    return can_be_outside and not proven_inside, can_be_inside


# Two values are the same array when they come from the same local or the same `new`
# instruction. slot is the local that holds value, for a value in the locals.
def same_array(array: Val, value: Val, slot: int | None = None) -> bool:
    same_local = known_equal(array.src, value.src) or known_equal(array.src, slot)
    same_allocation = known_equal(array.alloc_id, value.alloc_id)
    return same_local or same_allocation


# Library calls that cannot fail or loop so they add no outcomes.
SAFE_CALLS = ("Object.<init>", "AssertionError.<init>", "StringBuilder.")

@dataclass(frozen=True)
class State(sexpr.AsSExpr):
    locals: tuple[Val, ...]
    stack: tuple[Val, ...]

    def __str__(self) -> str:
        return f"{', '.join(map(str, self.locals))}/{':'.join(map(str, self.stack))}"

    def join(
        self,
        other: State,
        K: set[int] | list[int] | None = None,
        narrow: bool = False,
    ) -> State:
        if len(self.stack) != len(other.stack):
            raise AnalysisError(
                f"stack height mismatch: {len(self.stack)} vs {len(other.stack)}"
            )
        n_loc = max(len(self.locals), len(other.locals))
        l1 = self.locals + (Val.bot(),) * (n_loc - len(self.locals))
        l2 = other.locals + (Val.bot(),) * (n_loc - len(other.locals))
        return State(
            tuple(a.join(b, K, narrow) for a, b in zip(l1, l2)),
            tuple(a.join(b, K, narrow) for a, b in zip(self.stack, other.stack)),
        )

    def push(self, value: Val) -> State:
        return State(self.locals, self.stack + (value,))

    def pop(self, count: int = 1) -> tuple[tuple[Val, ...], State]:
        if count == 0:
            return (), self
        return self.stack[-count:], State(self.locals, self.stack[:-count])

    def load(self, idx: int) -> Val:
        return self.locals[idx] if idx < len(self.locals) else Val.top()

    def store(self, idx: int, value: Val, preserve_bound: bool = False) -> State:
        locs = list(self.locals)
        if idx >= len(locs):
            locs.extend([Val.bot()] * (idx - len(locs) + 1))
        locs[idx] = value
        new_locs = tuple(
            replace(
                entry,
                len_of=None if entry.len_of == idx else entry.len_of,
                bounded_by=entry.bounded_by
                if preserve_bound or entry.bounded_by != idx
                else None,
            )
            for entry in locs
        )
        stack = tuple(
            replace(
                entry,
                src=None if entry.src == idx else entry.src,
                len_of=None if entry.len_of == idx else entry.len_of,
                bounded_by=None if entry.bounded_by == idx else entry.bounded_by,
            )
            for entry in self.stack
        )
        return State(new_locs, stack)

    def forget_elements(self, refs: Iterable[Val]) -> State:
        # A called method can write into the arrays it gets. After the call their
        # elements can be anything.
        arrays = [ref for ref in refs if ref.is_ref]

        def forgotten(value: Val, slot: int | None = None) -> Val:
            if any(same_array(array, value, slot) for array in arrays):
                return replace(value, elem_iv=Interval.top())
            return value

        locs = tuple(forgotten(value, slot) for slot, value in enumerate(self.locals))
        stack = tuple(forgotten(value) for value in self.stack)
        return State(locs, stack)

    def update_array_elems(self, ref: Val, val_iv: Interval) -> State:
        def updated(value: Val, slot: int | None = None) -> Val:
            if same_array(ref, value, slot):
                return replace(
                    value,
                    may_null=False,
                    may_nonnull=True,
                    elem_iv=value.elem_iv | val_iv,
                    is_ref=True,
                )
            return value

        locs = tuple(updated(value, slot) for slot, value in enumerate(self.locals))
        stack = tuple(updated(value) for value in self.stack)
        return State(locs, stack)


class UnboundedIntervalAnalysis:
    def __init__(
        self,
        bc: jpamb.Bytecode,
        entry_method: jvm.AbsMethodID,
        inputs: jpamb.Input | None = None,
        param_ivs: dict[int, Interval] | None = None,
        domain: str = DEFAULT_DOMAIN,
    ):
        # Val takes the type of its pieces from Pieces, so the analysis sets it for
        # its domain before it builds any value.
        global Pieces
        Pieces = DOMAINS[domain]
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
                    case jvm.Push(value=int() as value) if not isinstance(value, bool):
                        ks |= {value - 1, value, value + 1}
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
                match opr:
                    case jvm.If() | jvm.Ifz() | jvm.Goto():
                        target = (pc % opr.target).offset
                        if target <= pc.offset:
                            hs.add(target)
            self.headers_cache[m] = hs
        return self.headers_cache[m]

    def _initial_state(self) -> State:
        method = self.bc.getmethod(self.entry_method)
        locs = [Val.bot()] * method.max_locals
        params = self.entry_method.extension.params

        if self.inputs is None:
            for i, p in enumerate(params):
                match p:
                    case jvm.Int():
                        locs[i] = Val(Interval.top())
                    case jvm.Short():
                        locs[i] = Val(SHORT)
                    case jvm.Byte():
                        locs[i] = Val(BYTE)
                    case jvm.Char():
                        locs[i] = Val(CHAR)
                    case jvm.Boolean():
                        locs[i] = Val(Interval(0, 1))
                    case jvm.Array(contains=jvm.Char()):
                        locs[i] = Val(
                            Interval.nonneg(),
                            may_null=True,
                            may_nonnull=True,
                            elem_iv=CHAR,
                            is_ref=True,
                        )
                    case jvm.Object(classname) if classname.dotted() == STRING:
                        locs[i] = Val(
                            Interval.nonneg(),
                            may_null=True,
                            may_nonnull=True,
                            elem_iv=CHAR,
                            is_ref=True,
                        )
                    case _:
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
                        elements = interval_of(ords)
                        locs[i] = Val(
                            Interval.const(len(vals)),
                            may_null=False,
                            may_nonnull=True,
                            elem_iv=elements,
                            is_ref=True,
                        )
                    case jpamb.case.Array(values=vals):
                        elements = interval_of(int(v) for v in vals)
                        locs[i] = Val(
                            Interval.const(len(vals)),
                            may_null=False,
                            may_nonnull=True,
                            elem_iv=elements,
                            is_ref=True,
                        )
                    case jpamb.case.String(value=s):
                        ords = [ord(c) for c in s]
                        elements = interval_of(ords)
                        locs[i] = Val(
                            Interval.const(len(s)),
                            may_null=False,
                            may_nonnull=True,
                            elem_iv=elements,
                            is_ref=True,
                        )
                    case _:
                        locs[i] = replace(Val.reference(), may_null=True)

        for i, iv in self.param_ivs.items():
            locs[i] = replace(locs[i], iv=locs[i].iv & iv)
        return State(tuple(locs), ())

    def run(self) -> UnboundedIntervalAnalysis:
        entry_pc = PC(self.entry_method, 0)
        init_st = self._initial_state()
        self.states = {entry_pc: init_st}
        work: list[tuple[str, int, PC]] = [(*pc_order(entry_pc), entry_pc)]
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
                    new = old.join(succ_st, K)
                    if new == old:
                        continue
                    self.states[succ_pc] = new

                if succ_pc not in queued:
                    queued.add(succ_pc)
                    heapq.heappush(work, (*pc_order(succ_pc), succ_pc))

        # Narrowing then takes back some of what widening added.
        for _ in range(2):
            incoming: dict[PC, State] = {entry_pc: init_st}
            for pc in sorted(self.states, key=pc_order):
                st = self.states[pc]
                for res in self.step(pc, st):
                    if isinstance(res, str):
                        self.outcomes.add(res)
                        continue
                    succ_pc, succ_st = res
                    if succ_pc in incoming:
                        incoming[succ_pc] = incoming[succ_pc].join(succ_st)
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

    def _wake_callers(
        self, callee: jvm.AbsMethodID, ret_val: Val
    ) -> list[tuple[PC, State]]:
        old_ret = self.return_vals.get(callee, Val.bot())
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
            receiver_count = 0 if isinstance(opr, jvm.InvokeStatic) else 1
            arg_count = len(callee.extension.params) + receiver_count
            args, after = caller_st.pop(arg_count)
            woken.append((call_pc + 1, after.forget_elements(args).push(joined_ret)))
        return woken

    # The successors of a call to a method in the bytecode, or None for a library
    # method which has no bytecode here.
    def _call(
        self, pc: PC, method: jvm.AbsMethodID, args: list[Val], after: State
    ) -> list[tuple[PC, State]] | None:
        try:
            callee = self.bc.getmethod(method)
        except (FileNotFoundError, IndexError):
            return None
        out = []
        if method.extension.return_type is None:
            out.append((pc + 1, after))
        else:
            self.call_sites.setdefault(method, set()).add(pc)
            if method in self.return_vals:
                out.append((pc + 1, after.push(self.return_vals[method])))
        locs = args + [Val.bot()] * (callee.max_locals - len(args))
        out.append((PC(method, 0), State(tuple(locs), ())))
        return out

    def step(self, pc: PC, state: State) -> list[tuple[PC, State] | str]:
        opr = self.bc[pc]
        out: list[tuple[PC, State] | str] = []

        match opr:
            case jvm.Push(type=t, value=v):
                match t:
                    case jvm.Int():
                        va = Val(Interval.const(int(v)))
                    case jvm.Reference() | jvm.Array():
                        va = Val.null()
                    case jvm.Object(classname) if classname.dotted() == STRING:
                        text = str(v)
                        ords = [ord(char) for char in text]
                        elements = interval_of(ords)
                        va = Val(
                            Interval.const(len(text)),
                            may_null=False,
                            may_nonnull=True,
                            elem_iv=elements,
                            is_ref=True,
                        )
                    case _:
                        if v is None:
                            va = Val.null()
                        elif isinstance(v, (bool, int)):
                            va = Val(Interval.const(int(v)))
                        else:
                            va = Val.reference()
                out.append((pc + 1, state.push(va)))

            case jvm.Load(index=i):
                out.append((pc + 1, state.push(replace(state.load(i), src=i))))

            case jvm.Store(index=i):
                [val], after = state.pop(1)
                out.append((pc + 1, after.store(i, replace(val, src=None))))

            case jvm.Incr(index=i, amount=a):
                v = state.load(i)
                step_pieces = type(v.piece_set).make([Interval.const(a)])
                _, added_pieces = pieces_arithmetic(
                    v.piece_set, step_pieces, jvm.BinaryOpr.Add
                )
                keep_bound = v.bounded_by if a <= 0 else None
                for result in v.iv.arithmetic(Interval.const(a), jvm.BinaryOpr.Add):
                    if isinstance(result, Interval):
                        new = Val(result, bounded_by=keep_bound, piece_set=added_pieces)
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
                neg_pieces = map_pieces(val.piece_set, negate_interval)
                out.append((pc + 1, after.push(Val(iv, piece_set=neg_pieces))))

            case jvm.Cast(to_=to):
                [val], after = state.pop(1)
                match to:
                    case jvm.Short():
                        target = SHORT
                    case jvm.Byte():
                        target = BYTE
                    case jvm.Char():
                        target = CHAR
                    case _:
                        target = Interval.top()
                iv = cast_interval(val.iv, target)
                cast_pieces = map_pieces(
                    val.piece_set, lambda piece: cast_interval(piece, target)
                )
                out.append((pc + 1, after.push(Val(iv, piece_set=cast_pieces))))

            case jvm.Binary(operant=op):
                [a, b], after = state.pop(2)
                results = a.iv.arithmetic(b.iv, op)
                piece_errors, value_pieces = pieces_arithmetic(
                    a.piece_set, b.piece_set, op
                )
                # An error needs both the interval and the pieces to allow it.
                errors = {result for result in results if isinstance(result, str)}
                out.extend(sorted(errors & piece_errors))
                values = [
                    result
                    for result in results
                    if isinstance(result, Interval) and not result.is_bot()
                ]
                if values:
                    joined = functools.reduce(operator.or_, values)
                    value = Val(joined, piece_set=value_pieces)
                    if not value.iv.is_bot():
                        out.append((pc + 1, after.push(value)))

            case jvm.Bitopr(operant=op):
                [a, b], after = state.pop(2)
                res_iv = a.iv.bitwise(b.iv, op)
                if not res_iv.is_bot():
                    out.append((pc + 1, after.push(Val(res_iv))))

            case jvm.Get(static=True, field=field):
                if field.extension.name == "$assertionsDisabled":
                    out.append((pc + 1, state.push(Val(Interval.const(0)))))
                else:
                    out.append((pc + 1, state.push(Val.top())))

            case jvm.New(classname=cls):
                if "AssertionError" in str(cls):
                    out.append("assertion error")
                out.append(
                    (
                        pc + 1,
                        state.push(
                            Val.reference()
                        ),
                    )
                )

            case jvm.NewArray(dim=d):
                sizes, after = state.pop(d)
                if sizes[0].iv.min < 0:
                    out.append("out of bounds")
                length_iv = sizes[0].iv & Interval.nonneg()
                if not length_iv.is_bot():
                    holds_elements = length_iv.max > 0
                    init_elem = Interval.const(0) if holds_elements else Interval.bot()
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
                    out.append((pc + 1, after.push(Val(length(ref), len_of=ref.src))))

            case jvm.ArrayLoad():
                [ref, idx], after = state.pop(2)
                if ref.may_null:
                    out.append("null pointer")
                if ref.may_nonnull:
                    outside, inside = check_index(ref, idx)
                    if outside:
                        out.append("out of bounds")
                    if inside:
                        elements = ref.elem_iv
                        loaded_iv = Interval.top() if elements.is_bot() else elements
                        out.append((pc + 1, after.push(Val(loaded_iv))))

            case jvm.ArrayStore():
                [ref, idx, val], after = state.pop(3)
                if ref.may_null:
                    out.append("null pointer")
                if ref.may_nonnull:
                    outside, inside = check_index(ref, idx)
                    if outside:
                        out.append("out of bounds")
                    if inside:
                        stored = after.update_array_elems(ref, val.iv)
                        out.append((pc + 1, stored))

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
                out.extend(
                    self._branch(pc, after, a, Val(Interval.const(0)), cond, target)
                )

            case jvm.If(condition=cond, target=target):
                [a, b], after = state.pop(2)
                out.extend(self._branch(pc, after, a, b, cond, target))

            case jvm.InvokeDynamic(method=m):
                nargs = len(m.extension.params)
                _, after = state.pop(nargs)
                returns_void = m.extension.return_type is None
                ret_val = Val(
                    Interval.nonneg(),
                    may_null=False,
                    may_nonnull=True,
                    elem_iv=CHAR,
                    is_ref=True,
                )
                out.append((pc + 1, after if returns_void else after.push(ret_val)))

            case jvm.InvokeStatic(method=m):
                nargs = len(m.extension.params)
                args, after = state.pop(nargs)
                after = after.forget_elements(args)
                called = self._call(pc, m, list(args), after)
                if called is None:
                    self.has_unknown_call = True
                    self.outcomes.update(ALL_OUTCOMES)
                    returns_void = m.extension.return_type is None
                    out.append(
                        (pc + 1, after if returns_void else after.push(Val.top()))
                    )
                else:
                    out.extend(called)

            case (
                jvm.InvokeVirtual(method=m)
                | jvm.InvokeSpecial(method=m)
                | jvm.InvokeInterface(method=m)
            ):
                nargs = len(m.extension.params) + 1
                args, after = state.pop(nargs)
                # A String cannot change, but getChars writes into a char[].
                if m.classname.dotted() == STRING:
                    after = after.forget_elements(args[1:])
                else:
                    after = after.forget_elements(args)
                recv_arg = args[0]
                if recv_arg.may_null:
                    out.append("null pointer")
                if recv_arg.may_nonnull:
                    returns_void = m.extension.return_type is None
                    after_any_result = after if returns_void else after.push(Val.top())
                    classname_str = m.classname.dotted()
                    method_name = m.extension.name

                    if classname_str == STRING:
                        match method_name:
                            case "equals":
                                out.append((pc + 1, after.push(Val(Interval(0, 1)))))
                            case "length":
                                size = Val(length(recv_arg), len_of=recv_arg.src)
                                out.append((pc + 1, after.push(size)))
                            case "charAt":
                                idx_arg = args[1]
                                str_len = length(recv_arg)
                                if idx_arg.iv.min < 0 or idx_arg.iv.max >= str_len.min:
                                    out.append("out of bounds")
                                letters = recv_arg.elem_iv
                                ch_iv = CHAR if letters.is_bot() else letters
                                out.append((pc + 1, after.push(Val(ch_iv))))
                            case _:
                                out.append((pc + 1, after_any_result))
                    elif classname_str == "jpamb.utils.Smith":
                        self.has_unknown_call = True
                        if method_name in ("nextBoolean", "flip"):
                            out.append((pc + 1, after.push(Val(Interval(0, 1)))))
                        else:
                            result = Val(Interval.top())
                            after_result = after if returns_void else after.push(result)
                            out.append((pc + 1, after_result))
                    else:
                        # The receiver is local 0 in the callee.
                        recv = Val(
                            recv_arg.iv,
                            src=0,
                            elem_iv=recv_arg.elem_iv,
                            is_ref=True,
                            piece_set=recv_arg.piece_set,
                        )
                        called = self._call(pc, m, [recv, *args[1:]], after)
                        if called is not None:
                            out.extend(called)
                        else:
                            if not any(name in str(m) for name in SAFE_CALLS):
                                self.has_unknown_call = True
                                self.outcomes.update(ALL_OUTCOMES)
                            out.append((pc + 1, after_any_result))

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
                                a.piece_set,
                            ),
                        )
                        if a.src is not None
                        else after
                    )
                res.append((tgt_pc, s))
            return res

        # if_acmpeq and if_acmpne compare references and give no int range.
        compares_references = a.may_null or b.may_null or a.is_ref or b.is_ref
        if isinstance(opr, jvm.If) and (reference_condition or compares_references):
            return [(pc % target, after), (pc + 1, after)]

        cmp_opr = cond if isinstance(cond, jvm.CmpOpr) else STR_TO_CMPOPR[cond_str]
        interval_outcomes = set(a.iv.compare(b.iv, cmp_opr))
        piece_outcomes = set(a.piece_set.compare(b.piece_set, cmp_opr))
        outcomes = interval_outcomes & piece_outcomes

        res = []
        for taken in (True, False):
            if taken not in outcomes:
                continue
            tgt_pc = (pc % target) if taken else (pc + 1)
            ra_iv, rb_iv = a.iv.refine(b.iv, cmp_opr, taken)
            ra_pieces, rb_pieces = a.piece_set.refine(b.piece_set, cmp_opr, taken)
            ra_iv, ra_pieces = reduce_pair(ra_iv, ra_pieces)
            rb_iv, rb_pieces = reduce_pair(rb_iv, rb_pieces)

            if ra_iv.is_bot() or rb_iv.is_bot():
                continue

            eff_opr = cmp_opr if taken else NEGATED[cmp_opr]

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
                        ra_pieces,
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
                        rb_pieces,
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
                only_null = val.may_null and not val.may_nonnull
                if val.is_ref or only_null or val.iv.is_bot():
                    continue
                # With the points 0 and 300 the fuzzer records 7 as the range [1, 299].
                # A sign stands for all ints of that sign so it is cut to the interval.
                for piece in val.piece_set.pieces:
                    for sub_iv in (piece & val.iv).split(pts):
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
        method_pcs = [pc for pc in before if pc.method == target_pc.method]
        at = len(method_pcs) - 1
        while at >= 0:
            match self.bc[method_pcs[at]]:
                case jvm.Store(index=local) if local == tracked_local:
                    # n = n op c is load n, push c, binary op and store n.
                    if at < 3:
                        return Interval.top()
                    update = [self.bc[pc] for pc in method_pcs[at - 3 : at]]
                    if not is_update_by_constant(*update, tracked_local):
                        return Interval.top()
                    _, push, binary = update
                    constant = Interval.const(push.value)
                    cur_iv = backward_arith_left(binary.operant, cur_iv, constant)
                    at -= 4
                    continue
                case jvm.Incr(index=local, amount=amount) if local == tracked_local:
                    step = Interval.const(amount)
                    cur_iv = backward_arith_left(jvm.BinaryOpr.Add, cur_iv, step)
            at -= 1

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
        cls,
        bc: jpamb.Bytecode,
        methodid: jvm.AbsMethodID,
        split: str = "three",
        domain: str = DEFAULT_DOMAIN,
    ) -> RangeCoverageOracle:
        analysis = UnboundedIntervalAnalysis(bc, methodid, None, domain=domain).run()
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

    last = jpamb.emit_init(analysis.states)
    for pc in sorted(analysis.states, key=lambda pc: pc.offset):
        if steps <= 0:
            break
        last = jpamb.emit_step(last, pc, analysis.states, depth=1)
        steps -= 1
    for outcome in sorted(analysis.outcomes):
        if steps <= 0:
            break
        jpamb.emit_step(last, PC(methodid, 0), outcome, depth=1)
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

    # When the analysis fails it answers maybe to every query.
    try:
        analysis = UnboundedIntervalAnalysis(bc, methodid, None).run()
        final = analysis.outcomes
        diverge = analysis.may_diverge()
        incomplete = False
    except Exception as error:
        traceback.print_exc()
        print(f"analysis failed: {error!r}", file=sys.stderr)
        final = set()
        diverge = True
        incomplete = True

    for query in jpamb.QUERIES:
        may_loop = query == "*" and diverge
        may = incomplete or query in final or may_loop
        print(f"{query};{'maybe' if may else 'no'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--interpret", action="store_true")
    args, _ = ap.parse_known_args()
    if args.interpret:
        interpret()
    else:
        analyse()

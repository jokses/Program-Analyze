#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import heapq
import sys
from dataclasses import dataclass
from typing import Optional

import jpamb
import sexpr
from jpamb import jvm
from jvm.state import PC
import traceback
INT_MIN, INT_MAX = -(2**31), 2**31 - 1
INT_TYPES = {"int", "i", "boolean", "z", "char", "c", "short", "s", "byte", "b"}

class AnalysisError(Exception):
    pass


# --------------------------------------------------------------------------
# Interval domain (with widening and narrowing)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Interval:
    lo: int
    hi: int

    @property
    def is_bot(self) -> bool:
        return self.lo > self.hi

    @property
    def single(self) -> bool:
        return self.lo == self.hi

    def __contains__(self, v: int) -> bool:
        return self.lo <= v <= self.hi

    def join(self, o: "Interval") -> "Interval":
        if self.is_bot:
            return o
        if o.is_bot:
            return self
        return Interval(min(self.lo, o.lo), max(self.hi, o.hi))

    def meet(self, o: "Interval") -> "Interval":
        lo, hi = max(self.lo, o.lo), min(self.hi, o.hi)
        return Interval(lo, hi) if lo <= hi else BOT

    def widen(self, new: "Interval", K: list[int]) -> "Interval":
        """K-widening: jump to the next constant in K instead of to infinity."""
        if self.is_bot:
            return new
        if new.is_bot:
            return self
        n = self.join(new)
        lo, hi = self.lo, self.hi
        if n.lo < lo:
            i = bisect.bisect_right(K, n.lo) - 1
            lo = K[i] if i >= 0 else INT_MIN
        if n.hi > hi:
            i = bisect.bisect_left(K, n.hi)
            hi = K[i] if i < len(K) else INT_MAX
        return Interval(lo, hi)

    def narrow(self, new: "Interval") -> "Interval":
        """Standard interval narrowing to recover precision after widening."""
        if self.is_bot or new.is_bot:
            return BOT
        lo = new.lo if self.lo == INT_MIN else self.lo
        hi = new.hi if self.hi == INT_MAX else self.hi
        return Interval(lo, hi) if lo <= hi else BOT

    def __str__(self) -> str:
        if self.is_bot:
            return "⊥"
        f = lambda v: "-inf" if v == INT_MIN else "+inf" if v == INT_MAX else str(v)
        return f"[{f(self.lo)},{f(self.hi)}]"


BOT = Interval(1, 0)
TOP = Interval(INT_MIN, INT_MAX)
NONNEG = Interval(0, INT_MAX)


def const(c: int) -> Interval:
    return Interval(c, c)


def clamp(lo: int, hi: int) -> Interval:
    """Java ints wrap; if a result can leave the int range, give up to TOP."""
    return TOP if lo < INT_MIN or hi > INT_MAX else Interval(lo, hi)


def _tdiv(x: int, y: int) -> int:
    q = abs(x) // abs(y)
    return -q if (x < 0) != (y < 0) else q


def arith(op: jvm.BinaryOpr, a: Interval, b: Interval) -> Interval:
    if a.is_bot or b.is_bot:
        return BOT
    match op:
        case jvm.BinaryOpr.Add:
            return clamp(a.lo + b.lo, a.hi + b.hi)
        case jvm.BinaryOpr.Sub:
            return clamp(a.lo - b.hi, a.hi - b.lo)
        case jvm.BinaryOpr.Mul:
            p = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
            return clamp(min(p), max(p))
        case jvm.BinaryOpr.Div:
            parts = []
            if b.lo <= -1:
                parts.append(b.meet(Interval(INT_MIN, -1)))
            if b.hi >= 1:
                parts.append(b.meet(Interval(1, INT_MAX)))
            qs = [_tdiv(x, y) for p in parts for x in (a.lo, a.hi) for y in (p.lo, p.hi)]
            return clamp(min(qs), max(qs)) if qs else BOT
        case jvm.BinaryOpr.Rem:
            if b.lo == 0 == b.hi:
                return BOT
            m = max(abs(b.lo), abs(b.hi))
            bound = max(abs(a.lo), abs(a.hi))
            lo = max(-(m - 1) if a.lo < 0 else 0, -bound)
            hi = min((m - 1) if a.hi > 0 else 0, bound)
            return Interval(lo, hi)
        case _:
            return TOP


def backward_arith_left(op: jvm.BinaryOpr, target: Interval, right: Interval) -> Interval:
    """Backward transfer for `left op right in target` (for uncovered range solving)."""
    if target.is_bot or right.is_bot:
        return BOT
    match op:
        case jvm.BinaryOpr.Add:
            return clamp(target.lo - right.hi, target.hi - right.lo)
        case jvm.BinaryOpr.Sub:
            return clamp(target.lo + right.lo, target.hi + right.hi)
        case jvm.BinaryOpr.Div:
            if right.single and right.lo > 0:
                d = right.lo
                lo = target.lo * d if target.lo >= 0 else (target.lo - 1) * d + 1
                hi = (target.hi + 1) * d - 1 if target.hi >= 0 else target.hi * d
                return clamp(lo, hi)
            return TOP
        case _:
            return TOP


NEG = {
    "eq": "ne",
    "ne": "eq",
    "lt": "ge",
    "ge": "lt",
    "gt": "le",
    "le": "gt",
    "is": "isnot",
    "isnot": "is",
}


def refine(a: Interval, b: Interval, cond: str) -> tuple[Interval, Interval]:
    """Narrow (a, b) assuming `a cond b` holds. May return BOT."""
    if cond == "eq":
        m = a.meet(b)
        return m, m
    if cond == "ne":
        a2, b2 = a, b
        if b.single:
            if a.lo == b.lo:
                a2 = Interval(a.lo + 1, a.hi) if a.lo < a.hi else BOT
            elif a.hi == b.lo:
                a2 = Interval(a.lo, a.hi - 1)
        if a.single:
            if b.lo == a.lo:
                b2 = Interval(b.lo + 1, b.hi) if b.lo < b.hi else BOT
            elif b.hi == a.lo:
                b2 = Interval(b.lo, b.hi - 1)
        return a2, b2
    if cond == "lt":
        return a.meet(Interval(INT_MIN, b.hi - 1)), b.meet(Interval(a.lo + 1, INT_MAX))
    if cond == "le":
        return a.meet(Interval(INT_MIN, b.hi)), b.meet(Interval(a.lo, INT_MAX))
    if cond == "gt":
        return a.meet(Interval(b.lo + 1, INT_MAX)), b.meet(Interval(INT_MIN, a.hi - 1))
    if cond == "ge":
        return a.meet(Interval(b.lo, INT_MAX)), b.meet(Interval(INT_MIN, a.hi))
    raise AnalysisError(f"unsupported condition {cond}")


# --------------------------------------------------------------------------
# Abstract Value & State (with array elements, nullness & length bounds)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Val:
    iv: Interval
    src: Optional[int] = None           # local index this value was loaded from
    may_null: bool = False              # reference may be null
    may_nonnull: bool = True            # reference may be non-null (iv = array length)
    elem_iv: Interval = TOP             # abstract interval of array elements
    len_of: Optional[int] = None        # local index of array whose length this value holds
    bounded_by: Optional[int] = None    # local index of array such that val < arr.length

    def join(self, o: "Val", K: Optional[list[int]] = None, narrow: bool = False) -> "Val":
        if narrow:
            iv = self.iv.narrow(o.iv)
            elem_iv = self.elem_iv.narrow(o.elem_iv)
        elif K is not None:
            iv = self.iv.widen(o.iv, K)
            elem_iv = self.elem_iv.widen(o.elem_iv, K)
        else:
            iv = self.iv.join(o.iv)
            elem_iv = self.elem_iv.join(o.elem_iv)

        return Val(
            iv=iv,
            src=self.src if self.src == o.src else None,
            may_null=self.may_null or o.may_null,
            may_nonnull=self.may_nonnull or o.may_nonnull,
            elem_iv=elem_iv,
            len_of=self.len_of if self.len_of == o.len_of else None,
            bounded_by=self.bounded_by if self.bounded_by == o.bounded_by else None,
        )

    def __str__(self) -> str:
        if self.may_null and not self.may_nonnull:
            return "null"
        prefix = "null|" if self.may_null else ""
        return f"{prefix}{self.iv}"


BOT_VAL = Val(BOT, None, False, False, BOT, None, None)
TOP_VAL = Val(TOP, None, True, True, TOP, None, None)


@dataclass(frozen=True)
class State(sexpr.AsSExpr):
    locals: tuple[Val, ...]
    stack: tuple[Val, ...]

    def __str__(self) -> str:
        return f"{', '.join(map(str, self.locals))}/{':'.join(map(str, self.stack))}"

    def join(self, other: "State", K: Optional[list[int]] = None, narrow: bool = False) -> "State":
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
        # Invalidate stack/local aliases referring to overwritten local `idx`
        new_locs = tuple(
            Val(
                x.iv,
                x.src,
                x.may_null,
                x.may_nonnull,
                x.elem_iv,
                None if x.len_of == idx else x.len_of,
                x.bounded_by if (preserve_bound or x.bounded_by != idx) else None,
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
            )
            for x in self.stack
        )
        return State(new_locs, stk)


# --------------------------------------------------------------------------
# High-Precision Unbounded Interval Analysis
# --------------------------------------------------------------------------
class UnboundedIntervalAnalysis:
    def __init__(self, bc: jpamb.Bytecode, entry_method: jvm.AbsMethodID, inputs: jpamb.Input | None = None):
        self.bc = bc
        self.entry_method = entry_method
        self.inputs = inputs
        self.K_cache: dict[jvm.AbsMethodID, list[int]] = {}
        self.headers_cache: dict[jvm.AbsMethodID, set[int]] = {}
        self.states: dict[PC, State] = {}
        self.outcomes: set[str] = set()
        self.preds: dict[PC, list[PC]] = {}

        # Interprocedural tracking
        self.return_vals: dict[jvm.AbsMethodID, Val] = {}
        self.call_sites: dict[jvm.AbsMethodID, set[PC]] = {}
        self.loop_edges: set[tuple[PC, PC]] = set()

    def _thresholds(self, m: jvm.AbsMethodID) -> list[int]:
        if m not in self.K_cache:
            ks = {INT_MIN, INT_MAX, -1, 0, 1}
            method = self.bc.getmethod(m)
            for opr in method.opcodes:
                match opr:
                    case jvm.Push(value=v):
                        raw = getattr(v, "value", v)
                        if isinstance(raw, int) and not isinstance(raw, bool):
                            ks |= {raw - 1, raw, raw + 1}
                    case jvm.Incr(amount=a):
                        ks.add(a)
            self.K_cache[m] = sorted(k for k in ks if INT_MIN <= k <= INT_MAX)
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
                p_str = str(p)
                if p_str in ("I", "int"):
                    locs[i] = Val(TOP)
                elif p_str in ("S", "short"):
                    locs[i] = Val(Interval(-32768, 32767))
                elif p_str in ("B", "byte"):
                    locs[i] = Val(Interval(-128, 127))
                elif p_str in ("C", "char"):
                    locs[i] = Val(Interval(0, 65535))
                elif p_str in ("Z", "boolean"):
                    locs[i] = Val(Interval(0, 1))
                elif p_str.startswith("["):
                    locs[i] = Val(NONNEG, may_null=True, may_nonnull=True, elem_iv=TOP)
                else:
                    locs[i] = TOP_VAL
        else:
            for i, x in enumerate(self.inputs.values):
                match x:
                    case jpamb.case.Boolean(value=v) | jpamb.case.Int(value=v):
                        locs[i] = Val(const(int(v)))
                    case jpamb.case.Array(values=vals):
                        if vals:
                            e_lo = min(int(v) for v in vals)
                            e_hi = max(int(v) for v in vals)
                            e_iv = Interval(e_lo, e_hi)
                        else:
                            e_iv = BOT
                        locs[i] = Val(const(len(vals)), may_null=False, may_nonnull=True, elem_iv=e_iv)
                    case _:
                        locs[i] = TOP_VAL

        return State(tuple(locs), ())

    def run(self) -> "UnboundedIntervalAnalysis":
        entry_pc = PC(self.entry_method, 0)
        self.states = {entry_pc: self._initial_state()}
        work: list[tuple[str, int, PC]] = [(str(entry_pc.method), entry_pc.offset, entry_pc)]
        queued: set[PC] = {entry_pc}

        # 1. Ascending fixpoint with K-widening
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
                    K = self._thresholds(succ_pc.method) if succ_pc.offset in headers else None
                    new = old.join(succ_st, K, narrow=False)
                    if new == old:
                        continue
                    self.states[succ_pc] = new

                if succ_pc not in queued:
                    queued.add(succ_pc)
                    heapq.heappush(work, (str(succ_pc.method), succ_pc.offset, succ_pc))

        # 2. Descending narrowing pass (2 iterations over loop headers to tighten widened bounds)
        for _ in range(2):
            for pc in sorted(self.states.keys(), key=lambda p: (str(p.method), p.offset)):
                st = self.states[pc]
                for res in self.step(pc, st):
                    if isinstance(res, str):
                        continue
                    succ_pc, succ_st = res
                    if succ_pc in self.states:
                        narrowed = self.states[succ_pc].join(succ_st, narrow=True)
                        if narrowed != self.states[succ_pc] and not any(v.iv.is_bot and v.may_nonnull for v in narrowed.locals):
                            self.states[succ_pc] = narrowed

        return self

    def may_diverge(self) -> bool:
        """
        Determine whether non-termination ('*') is genuinely possible.
        A loop that has a reachable exit edge and bounded monotonic induction variables terminates.
        """
        if not self.loop_edges:
            return False

        for tail_pc, header_pc in self.loop_edges:
            m = header_pc.method
            method = self.bc.getmethod(m)
            # Check if every instruction inside the loop body can reach an exit outside [header, tail]
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
                        if tgt.method == m and (tgt.offset < header_pc.offset or tgt.offset > tail_pc.offset):
                            has_reachable_exit = True

            if not has_reachable_exit:
                # Infinite loop with no reachable exit (e.g., while(true) {})
                return True

            # Check if the exit branch depends on a variable widened to +/-inf
            header_st = self.states.get(header_pc)
            if header_st is not None:
                for lpc in loop_pcs:
                    opr = self.bc[lpc]
                    if isinstance(opr, (jvm.If, jvm.Ifz)):
                        st_at_branch = self.states.get(lpc)
                        if st_at_branch and st_at_branch.stack:
                            top_iv = st_at_branch.stack[-1].iv
                            # If loop condition operand hit +/-inf, loop bound is unknown -> may diverge
                            if top_iv.lo == INT_MIN or top_iv.hi == INT_MAX:
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
            _, after = caller_st.pop(nargs)
            woken.append((call_pc + 1, after.push(joined_ret)))
        return woken

    def step(self, pc: PC, state: State) -> list[tuple[PC, State] | str]:
        opr = self.bc[pc]
        out: list[tuple[PC, State] | str] = []

        match opr:
            case jvm.Push(value=v):
                raw = getattr(v, "value", v)
                if str(opr).startswith("push:A") or raw is None:
                    va = Val(BOT, may_null=True, may_nonnull=False)
                elif isinstance(raw, (bool, int)):
                    va = Val(const(int(raw)))
                else:
                    va = TOP_VAL
                out.append((pc + 1, state.push(va)))

            case jvm.Load(index=i):
                v = state.load(i)
                out.append((pc + 1, state.push(Val(v.iv, i, v.may_null, v.may_nonnull, v.elem_iv, v.len_of, v.bounded_by))))

            case jvm.Store(index=i):
                [val], after = state.pop(1)
                stored = Val(val.iv, None, val.may_null, val.may_nonnull, val.elem_iv, val.len_of, val.bounded_by)
                out.append((pc + 1, after.store(i, stored)))

            case jvm.Incr(index=i, amount=a):
                v = state.load(i)
                res = arith(jvm.BinaryOpr.Add, v.iv, const(a))
                # Incrementing an index may break upper bound (< arr.length), unless a <= 0
                keep_bound = v.bounded_by if a <= 0 else None
                out.append((pc + 1, state.store(i, Val(res, bounded_by=keep_bound))))

            case jvm.Dup():
                [val], _ = state.pop(1)
                out.append((pc + 1, state.push(val)))

            case jvm.Negate():
                [val], after = state.pop(1)
                iv = clamp(-val.iv.hi, -val.iv.lo) if not val.iv.is_bot else BOT
                out.append((pc + 1, after.push(Val(iv))))

            case jvm.Cast():
                [val], after = state.pop(1)
                out.append((pc + 1, after.push(Val(val.iv))))

            case jvm.Binary(operant=op):
                [a, b], after = state.pop(2)
                if op in (jvm.BinaryOpr.Div, jvm.BinaryOpr.Rem) and 0 in b.iv:
                    out.append("divide by zero")
                r = arith(op, a.iv, b.iv)
                if not r.is_bot:
                    out.append((pc + 1, after.push(Val(r))))

            case jvm.Get(static=True, field=field):
                if getattr(field.extension, "name", "") == "$assertionsDisabled":
                    out.append((pc + 1, state.push(Val(const(0)))))
                else:
                    out.append((pc + 1, state.push(TOP_VAL)))

            case jvm.New(classname=cls):
                if "AssertionError" in str(cls):
                    out.append("assertion error")
                out.append((pc + 1, state.push(Val(TOP, may_null=False, may_nonnull=True))))

            case jvm.NewArray(dim=d):
                sizes, after = state.pop(d)
                length_iv = sizes[0].iv.meet(NONNEG)
                if not length_iv.is_bot:
                    # Fresh Java arrays are zero-initialized (elem_iv = [0, 0] if length > 0)
                    init_elem = const(0) if length_iv.hi > 0 else BOT
                    out.append((pc + 1, after.push(Val(length_iv, may_null=False, may_nonnull=True, elem_iv=init_elem))))

            case jvm.ArrayLength():
                [ref], after = state.pop(1)
                if ref.may_null:
                    out.append("null pointer")
                if ref.may_nonnull:
                    len_iv = ref.iv.meet(NONNEG) if not ref.iv.is_bot else NONNEG
                    out.append((pc + 1, after.push(Val(len_iv, len_of=ref.src))))

            case jvm.ArrayLoad():
                [ref, idx], after = state.pop(2)
                if ref.may_null:
                    out.append("null pointer")
                if ref.may_nonnull:
                    arr_len = ref.iv.meet(NONNEG) if not ref.iv.is_bot else NONNEG
                    is_provably_in_bounds = (
                        ref.src is not None
                        and idx.bounded_by == ref.src
                        and idx.iv.lo >= 0
                    )
                    if not is_provably_in_bounds:
                        if idx.iv.lo < 0 or idx.iv.hi >= arr_len.lo:
                            out.append("out of bounds")
                    valid_idx = idx.iv.meet(Interval(0, arr_len.hi - 1)) if arr_len.hi > 0 else BOT
                    if not valid_idx.is_bot:
                        loaded_iv = ref.elem_iv if not ref.elem_iv.is_bot else TOP
                        out.append((pc + 1, after.push(Val(loaded_iv))))

            case jvm.ArrayStore():
                [ref, idx, val], after = state.pop(3)
                if ref.may_null:
                    out.append("null pointer")
                if ref.may_nonnull:
                    arr_len = ref.iv.meet(NONNEG) if not ref.iv.is_bot else NONNEG
                    is_provably_in_bounds = (
                        ref.src is not None
                        and idx.bounded_by == ref.src
                        and idx.iv.lo >= 0
                    )
                    if not is_provably_in_bounds:
                        if idx.iv.lo < 0 or idx.iv.hi >= arr_len.lo:
                            out.append("out of bounds")
                    valid_idx = idx.iv.meet(Interval(0, arr_len.hi - 1)) if arr_len.hi > 0 else BOT
                    if not valid_idx.is_bot:
                        next_st = after
                        if ref.src is not None:
                            old_arr = next_st.load(ref.src)
                            new_elem = old_arr.elem_iv.join(val.iv)
                            updated_arr = Val(
                                old_arr.iv,
                                old_arr.src,
                                False,
                                True,
                                new_elem,
                                old_arr.len_of,
                                old_arr.bounded_by,
                            )
                            next_st = next_st.store(ref.src, updated_arr, preserve_bound=True)
                        out.append((pc + 1, next_st))

            case jvm.Goto(target=t):
                out.append((pc % t, state))

            case jvm.Return(type=ret_type):
                out.append("ok")
                if ret_type is not None and state.stack:
                    out.extend(self._wake_callers(pc.method, state.stack[-1]))

            case jvm.Throw():
                pass

            case jvm.Ifz(condition=cond, target=target):
                [a], after = state.pop(1)
                out.extend(self._branch(pc, after, a, Val(const(0)), cond, target))

            case jvm.If(condition=cond, target=target):
                [a, b], after = state.pop(2)
                out.extend(self._branch(pc, after, a, b, cond, target))

            case jvm.InvokeStatic(method=m):
                nargs = len(m.extension.params)
                args, after = state.pop(nargs)
                returns_void = m.extension.return_type is None or str(m).endswith("V")
                if returns_void:
                    out.append((pc + 1, after))
                else:
                    self.call_sites.setdefault(m, set()).add(pc)
                    if m in self.return_vals:
                        out.append((pc + 1, after.push(self.return_vals[m])))
                    elif m not in self.bc.methods:
                        out.append((pc + 1, after.push(TOP_VAL)))
                try:
                    callee = self.bc.getmethod(m)
                    locs = list(args) + [BOT_VAL] * (callee.max_locals - nargs)
                    out.append((PC(m, 0), State(tuple(locs), ())))
                except Exception:
                    if not returns_void:
                        out.append((pc + 1, after.push(TOP_VAL)))

            case (
                jvm.InvokeVirtual(method=m)
                | jvm.InvokeSpecial(method=m)
                | jvm.InvokeInterface(method=m)
            ):
                nargs = len(m.extension.params) + 1
                args, after = state.pop(nargs)
                if args[0].may_null:
                    out.append("null pointer")
                if args[0].may_nonnull:
                    returns_void = m.extension.return_type is None or str(m).endswith("V")
                    if returns_void:
                        out.append((pc + 1, after))
                    else:
                        self.call_sites.setdefault(m, set()).add(pc)
                        if m in self.return_vals:
                            out.append((pc + 1, after.push(self.return_vals[m])))
                        else:
                            out.append((pc + 1, after.push(TOP_VAL)))
                    try:
                        callee = self.bc.getmethod(m)
                        recv = Val(args[0].iv, 0, False, True, args[0].elem_iv)
                        locs = [recv] + list(args[1:]) + [BOT_VAL] * (callee.max_locals - nargs)
                        out.append((PC(m, 0), State(tuple(locs), ())))
                    except Exception:
                        pass

            case _:
                raise NotImplementedError(f"Unsupported operation {opr!r} at {pc}")

        return out
    def _is_ref_test(self, pc: PC, a: Val) -> bool:
        """ifz on a reference is a null test (ifnull / ifnonnull), not `== 0`."""
        if a.may_null:
            return True
        prev = self.bc[PC(pc.method, pc.offset - 1)]
        if isinstance(prev, jvm.Load):
            return str(prev.type).lower() not in INT_TYPES
        return False
    def _branch(
        self, pc: PC, after: State, a: Val, b: Val, cond, target: int
    ) -> list[tuple[PC, State]]:
        cond = str(cond)  # CmpOpr.Ne -> "ne"
        opr = self.bc[pc]

        unknown = False  # reference comparison we can't refine: both edges feasible
        if isinstance(opr, jvm.Ifz) and cond in ("eq", "ne") and self._is_ref_test(pc, a):
            cond = "is" if cond == "eq" else "isnot"
        elif isinstance(opr, jvm.If) and (a.may_null or b.may_null):
            unknown = True

        res: list[tuple[PC, State]] = []
        for taken, c in ((True, cond), (False, NEG.get(cond))):
            tgt_pc = (pc % target) if taken else (pc + 1)
            if unknown or c is None:
                res.append((tgt_pc, after))
                continue

            if c in ("is", "isnot"):
                if c == "is":
                    if not a.may_null:
                        continue
                    s = after.store(a.src, Val(BOT, a.src, True, False)) if a.src is not None else after
                else:
                    if not a.may_nonnull:
                        continue
                    s = (
                        after.store(a.src, Val(a.iv, a.src, False, True, a.elem_iv, a.len_of, a.bounded_by))
                        if a.src is not None
                        else after
                    )
                res.append((tgt_pc, s))
                continue

            ra, rb = refine(a.iv, b.iv, c)
            if ra.is_bot or rb.is_bot:
                continue

            s = after
            a_bound = b.len_of if (c == "lt" and b.len_of is not None) else a.bounded_by
            b_bound = a.len_of if (c == "gt" and a.len_of is not None) else b.bounded_by

            if a.src is not None:
                s = s.store(
                    a.src,
                    Val(ra, a.src, a.may_null, a.may_nonnull, a.elem_iv, a.len_of, a_bound),
                    preserve_bound=True,
                )
            if b.src is not None:
                s = s.store(
                    b.src,
                    Val(rb, b.src, b.may_null, b.may_nonnull, b.elem_iv, b.len_of, b_bound),
                    preserve_bound=True,
                )
            res.append((tgt_pc, s))
        return res

    # -- Fuzzer Range Coverage Cells & Backward Refinement (Proposal) -------
    def cells(self, pc: PC) -> list[tuple[str, Interval]]:
        """Possible ranges at an if/ifz: the bound the fuzzer's coverage meets."""
        st = self.states.get(pc)
        if st is None:
            return []
        opr = self.bc[pc]
        match opr:
            case jvm.Ifz(condition=cond) if cond not in ("is", "isnot"):
                a, b = st.stack[-1].iv, const(0)
            case jvm.If(condition=cond) if cond not in ("is", "isnot"):
                a, b = st.stack[-2].iv, st.stack[-1].iv
            case _:
                return []

        pivot = str(b.lo) if b.single else "y"
        parts = [
            (f"x<{pivot}", a.meet(Interval(INT_MIN, b.hi - 1))),
            (f"x=={pivot}", a.meet(b)),
            (f"x>{pivot}", a.meet(Interval(b.lo + 1, INT_MAX))),
        ]
        return [(lbl, iv) for lbl, iv in parts if not iv.is_bot]

    def backward_refine(self, target_pc: PC, local_idx: int, target_iv: Interval) -> Interval:
        """Backward analysis from an uncovered target range to narrow parameter inputs."""
        cur_iv = target_iv
        visited: set[PC] = set()
        curr = target_pc
        while curr.offset > 0 and curr not in visited:
            visited.add(curr)
            preds = self.preds.get(curr, [])
            if not preds:
                break
            prev = next((p for p in preds if p.offset == curr.offset - 1), preds[0])
            opr = self.bc[prev]
            match opr:
                case jvm.Store(index=i) if i == local_idx:
                    if prev.offset >= 2:
                        bin_pc = PC(prev.method, prev.offset - 1)
                        push_pc = PC(prev.method, prev.offset - 2)
                        bin_opr = self.bc[bin_pc]
                        push_opr = self.bc[push_pc]
                        if isinstance(bin_opr, jvm.Binary) and isinstance(push_opr, jvm.Push):
                            c = getattr(push_opr.value, "value", push_opr.value)
                            if isinstance(c, int):
                                cur_iv = backward_arith_left(bin_opr.operant, cur_iv, const(c))
                                curr = push_pc
                                continue
                case jvm.Incr(index=i, amount=a) if i == local_idx:
                    cur_iv = backward_arith_left(jvm.BinaryOpr.Add, cur_iv, const(a))
            curr = prev
        return cur_iv


# --------------------------------------------------------------------------
# JPAMB Entry Points (analyse / interpret)
# --------------------------------------------------------------------------
def interpret():
    """JPAMB interpreter mode for step-by-step abstract state emission."""
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
    """JPAMB static analysis entry point (`jpamb analyse ... unbounded`)."""
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
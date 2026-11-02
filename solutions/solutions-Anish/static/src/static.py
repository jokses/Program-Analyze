import sys
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from abstractions import SignSet

import jpamb
import sexpr
from jpamb import jvm
from jvm.state import PC, StackInt


@dataclass(frozen=True)
class State(sexpr.AsSExpr):
    locals: tuple[SignSet, ...]
    stack: tuple[SignSet, ...]

    def __post_init__(self):
        assert isinstance(self.locals, tuple)
        assert isinstance(self.stack, tuple)

    def __str__(self):
        return f"{', '.join(map(str, self.locals))}/{':'.join(map(str, self.stack))}"

    def __or__(self, other):
        assert isinstance(other, State), f"Expected State but got {other!r}"
        assert len(self.stack) == len(other.stack), "Stacks should be equal lenght"
        assert len(self.locals) == len(other.locals), "Locals should be equal lenght"

        return State(
            tuple(s1 | s2 for s1, s2 in zip(self.locals, other.locals)),
            tuple(s1 | s2 for s1, s2 in zip(self.stack, other.stack)),
        )

    def push(self, value: SignSet):
        assert isinstance(value, SignSet), f"Expected sign set but got {value}"
        return State(self.locals, self.stack + (value,))

    def pop(self, number=1):
        return self.stack[-number:], State(self.locals, self.stack[:-number])

    def load(self, index):
        return self.locals[index]

    def store(self, index, value):
        return State(
            tuple(self.locals[:index]) + (value,) + tuple(self.locals[index + 1 :]),
            self.stack,
        )


def manystep(
    bc: jpamb.Bytecode,
    pc: PC,
    state: State,
) -> Iterable[tuple[PC, object] | str]:
    opr = bc[pc]
    match opr:
        case jvm.Get(static=True, field=field):
            # Hack - Only handle the assertion case
            assert field.extension.name == "$assertionsDisabled"

            # Hack - Assuming assertions are never disabled
            va = SignSet.abstract([StackInt(0)])

            yield (pc + 1, state.push(va))

        case jvm.Ifz(condition=op, target=target):
            [val], after = state.pop(1)

            for res in SignSet.compare(val, SignSet.abstract([StackInt(0)]), op):
                match res:
                    case True:
                        yield (pc % target, after)
                    case False:
                        yield (pc + 1, after)
                    case err:
                        yield err

        case jvm.Load(index=i):
            va = state.load(i)
            yield (pc + 1, state.push(va))

        case jvm.Goto(target=t):
            yield (pc % t, state)

        case jvm.Binary(operant=op):
            [v1, v2], after = state.pop(2)
            result, errors = v1.arithmetic(v2, op)
            for err in errors:
                yield err
            if result.signs:  # an empty result means every path errored
                yield (pc + 1, after.push(result))

        case jvm.Return(type=None):
            yield "ok"

        case jvm.Return(type=t):
            # Hack -- we assume that we always return.
            yield "ok"

        case jvm.New(classname=jvm.ClassName("java.lang.AssertionError")):
            # Hack -- if we create an assertion error, we probably also throw it.
            yield "assertion error"
        case jvm.Push(value=v):
            raw = getattr(v, "value", v)
            if str(opr).startswith("push:A"):
                va = SignSet(frozenset({-1}))            # null reference
            elif isinstance(raw, (bool, int)):
                va = SignSet.abstract([StackInt(int(raw))])
            else:
                va = SignSet.top()
            yield (pc + 1, state.push(va))
        case jpamb.case.Array():
            locals[i] = SignSet(frozenset({0, 1}))
        case jvm.Store(index=i):
            [val], after = state.pop(1)
            yield (pc + 1, after.store(i, val))

        case jvm.Incr(index=i, amount=a):
            one = SignSet.abstract([StackInt(a)])
            result, _ = state.load(i).arithmetic(one, jvm.BinaryOpr.Add)
            yield (pc + 1, state.store(i, result))

        case jvm.Dup():
            [val], _ = state.pop(1)
            yield (pc + 1, state.push(val))


        case jvm.If(condition=op, target=target):
            [v1, v2], after = state.pop(2)
            for res in v1.compare(v2, op):
                if res:
                    yield (pc % target, after)
                else:
                    yield (pc + 1, after)
        case jvm.NewArray(dim=d):
            sizes, after = state.pop(d)
            # a negative size would throw NegativeArraySizeException, which is not an outcome we track
            ok_signs = frozenset(s for s in sizes[0].signs if s >= 0)
            if ok_signs:
                yield (pc + 1, after.push(SignSet(ok_signs)))

        case jvm.ArrayLength():
            [ref], after = state.pop(1)
            if -1 in ref.signs:
                yield "null pointer"
            lengths = frozenset(s for s in ref.signs if s >= 0)
            if lengths:
                yield (pc + 1, after.push(SignSet(lengths)))

        case jvm.ArrayLoad():
            [ref, idx], after = state.pop(2)
            if -1 in ref.signs:
                yield "null pointer"
            lengths = {s for s in ref.signs if s >= 0}
            if lengths:
                # out of bounds is possible if the index may be negative, the array may be
                # empty, or the index may be positive (we cannot compare it with the length)
                if -1 in idx.signs or 0 in lengths or 1 in idx.signs:
                    yield "out of bounds"
                if 1 in lengths and (0 in idx.signs or 1 in idx.signs):
                    yield (pc + 1, after.push(SignSet.top()))

        case jvm.ArrayStore():
            [ref, idx, _val], after = state.pop(3)
            if -1 in ref.signs:
                yield "null pointer"
            lengths = {s for s in ref.signs if s >= 0}
            if lengths:
                if -1 in idx.signs or 0 in lengths or 1 in idx.signs:
                    yield "out of bounds"
                if 1 in lengths and (0 in idx.signs or 1 in idx.signs):
                    yield (pc + 1, after)

        case jvm.Negate():
            [val], after = state.pop(1)
            yield (pc + 1, after.push(SignSet(frozenset(-s for s in val.signs))))

        case jvm.Cast():
            [_val], after = state.pop(1)
            yield (pc + 1, after.push(SignSet.top()))

        case jvm.Throw():
            # an exception we cannot classify ends the method; the `New` case already
            # reports `assertion error` when one is created
            return
        case jpamb.case.Array():
            locals[i] = SignSet(frozenset({0, 1}))
        case jvm.InvokeStatic(method=m):
            nargs = len(m.extension.params)
            if nargs:
                args, after = state.pop(nargs)   # guard: pop(0) would return the whole stack
            else:
                args, after = (), state

            # 1. the caller continues after the call (pushing an unknown result unless void)
            returns_void = str(m).endswith("V")   # hack: see the note below
            yield (pc + 1, after if returns_void else after.push(SignSet.top()))

            # 2. the callee starts running with the arguments as its first locals
            try:
                callee = bc.getmethod(m)
                locs = list(args) + [SignSet.bot()] * (callee.max_locals - nargs)
                yield (PC(m, 0), State(tuple(locs), ()))
            except Exception:
                pass   # a method we cannot look up (for example a library call)
        case (
            jvm.InvokeVirtual(method=m)
            | jvm.InvokeSpecial(method=m)
            | jvm.InvokeInterface(method=m)
        ):
            nargs = len(m.extension.params) + 1      # arguments plus the receiver
            _, after = state.pop(nargs)
            if m.extension.return_type is None:      # void (assumed to be None, see below)
                yield (pc + 1, after)
            else:
                yield (pc + 1, after.push(SignSet.top()))
        case _:
            raise NotImplementedError(f"Unsupported operation {opr!r}")         


def initialstate(
    bc: jpamb.Bytecode,
    methodid: jvm.AbsMethodID,
    inputs: jpamb.Input | None,
) -> dict[PC, State]:
    method = bc.getmethod(methodid)
    locals = [SignSet.bot()] * method.max_locals

    if inputs is None:
        for i, p in enumerate(methodid.extension.params):
            locals[i] = SignSet.top()
    else:
        for i, x in enumerate(inputs.values):
            match x:
                case jpamb.case.Boolean(value=value):
                    locals[i] = SignSet.abstract([StackInt(int(value))])
                case jpamb.case.Int(value=value):
                    locals[i] = SignSet.abstract([StackInt(int(value))])
                case jpamb.case.Array():
                    locals[i] = SignSet.from_sign("+")
                case _:
                    locals[i] = SignSet.top()

    state = State(tuple(locals), ())
    return {PC(methodid, 0): state}


@dataclass
class AbstractInterpreter:
    bc: jpamb.Bytecode
    worklist: deque[PC]
    states: dict[PC, State]
    loop_seen: bool = False
    @staticmethod
    def initial(bc: jpamb.Bytecode, methodid: jvm.AbsMethodID, inputs):
        states = initialstate(bc, methodid, inputs)
        worklist = deque(states.keys())

        return AbstractInterpreter(bc, worklist, states)

    def step(self) -> tuple[PC, set[str]]:
        pc = self.worklist.pop()
        

        print(f"Stepping {pc}:\n > {self.bc[pc]}", file=sys.stderr)

        finals = set()

        for res in manystep(self.bc, pc, self.states[pc]):
            if isinstance(res, str):
                finals.add(res)
            else:
                pc_, st = res
                if pc_.method == pc.method and pc_.offset <= pc.offset:
                    self.loop_seen = True   # a backward jump within one method: possible loop
                before = self.states.get(pc_, None)
                if before is None:
                    after = st
                else:
                    after = before | st
                if before is None or after != before:
                    self.states[pc_] = after
                    self.worklist.append(pc_)

        return pc, finals


def interpret():
    """The static analysis"""
    methodid, input, steps = jpamb.getcase(
        "static",
        "1.0",
        "kindleframe",
        ["static", "python"],
        for_science=True,
    )
    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(suite, eff, {})

    ai = AbstractInterpreter.initial(bc, methodid, input)

    x = jpamb.emit_init(ai.states)

    while steps > 0 and ai.worklist:
        seen_final = False
        last_pc = None
        while steps > 0 and ai.worklist:
            pc, final = ai.step()
            last_pc = pc
            for f in final:
                seen_final = True
                jpamb.emit_step(x, pc, f, depth=1)
                steps -= 1

            x = jpamb.emit_step(x, pc, ai.states, depth=1)
            steps -= 1

        # No path ever finished: the method may run forever.
        if last_pc is not None and not seen_final:
            jpamb.emit_step(x, last_pc, "*", depth=1)

def analyse():
    """The static analysis"""
    methodid = jpamb.getmethodid(
        "static",
        "1.0",
        "kindleframe",
        ["static", "python"],
        for_science=True,
    )
    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(suite, eff, {})

    steps = 5000
    final: set[str] = set()
    incomplete = False
    loop = False
    try:
        ai = AbstractInterpreter.initial(bc, methodid, None)
        while steps > 0 and ai.worklist:
            _pc, finals = ai.step()
            final |= finals
            steps -= 1
        incomplete = bool(ai.worklist)
        loop = ai.loop_seen
    except Exception as e:
        print(f"analysis failed: {e!r}", file=sys.stderr)
        incomplete = True

    for f in jpamb.QUERIES:
        may = incomplete or f in final or (f == "*" and loop)
        print(f"{f};{'maybe' if may else 'no'}")

import random
import sys
from wsgiref.validate import assert_

import jpamb
import jvm
import jvm.state as jvmc


def binary(op, v1: int, v2: int) -> int | str:
    match op:
        case jvm.BinaryOpr.Div:
            try:
                return v1 // v2
            except ZeroDivisionError:
                return "divide by zero"
        case jvm.BinaryOpr.Sub:
            return v1 - v2
        case jvm.BinaryOpr.Mul:
            return v1 * v2
        case jvm.BinaryOpr.Add:
            return v1 + v2
        case a:
            raise NotImplementedError(f"Unhandled binary {op!r}")


def compare(op, v1: int, v2: int) -> bool:
    match op:
        case jvm.CmpOpr.Eq:
            return v1 == v2
        case jvm.CmpOpr.Ne:
            return v1 != v2
        case jvm.CmpOpr.Lt:
            return v1 < v2
        case jvm.CmpOpr.Le:
            return v1 <= v2
        case jvm.CmpOpr.Gt:
            return v1 > v2
        case jvm.CmpOpr.Ge:
            return v1 >= v2
        case _:
            raise NotImplementedError(f"Unhandled comparation {op!r}")


def step(bc: jpamb.Bytecode, state: jvmc.State) -> tuple[jvmc.PC, jvmc.State | str]:
    assert isinstance(state, jvmc.State), f"expected state but got {state}"
    frame = state.frames.peek()
    pc = frame.pc
    opr = bc[pc]
    output = state
    print(f"Stepping {pc}:\n > {opr}", file=sys.stderr)
    match opr:
        case jvm.Push(type=t, value=v):
            match t:
                case jvm.Int():
                    frame.stack.push(jvmc.StackInt(v))
                # case :
                #     frame.stack.push(jvmc.StackRef(v))
                case _:
                    raise NotImplementedError(f"expected int, but got {jvm.Int()} {t} {t!r}")
            frame.pc += 1

        case jvm.Ifz(offset= off , condition = cond , target= targ):
            value = frame.stack.pop()
            assert isinstance(value, jvmc.StackInt), f"expected int, but got {value}"

            # print(f" value  -----  {type(frame.pc.offset)} {cond}  {value} {type(targ)}", file=sys.stderr)
            if compare(cond,value.value, 0):
                frame.pc %= targ
            else:
                frame.pc +=1
        case jvm.Load( index= index):
            print(f" value ----  {index}", file=sys.stderr)

            frame.stack.push(frame.locals[index])
            frame.pc +=1

        case jvm.Binary(type=jvm.Int(), operant=op):
            v2, v1 = frame.stack.pop(), frame.stack.pop()
            assert isinstance(v1, jvmc.StackInt), f"expected int, but got {v1}"
            assert isinstance(v2, jvmc.StackInt), f"expected int, but got {v2}"

            value = binary(op, v1.value, v2.value)

            if isinstance(value, str):
                output = value
            else:
                frame.stack.push(jvmc.StackInt(value))
                frame.pc += 1

        case jvm.Return(type=t):
            match t:
                case jvm.Int():
                    v1 = frame.stack.pop()
                    state.frames.pop()
                    if state.frames:
                        frame = state.frames.peek()
                        frame.stack.push(v1)
                        frame.pc += 1
                    else:
                        output = "ok"
                case None:
                    output = "ok"
                case _:
                    raise NotImplementedError(f"Unhandled binary {t} {t!r}")

        case jvm.If(offset=off,condition=cond, target=targ):
            value2 = frame.stack.pop()
            value1 = frame.stack.pop()
            assert isinstance(value1, jvmc.StackInt), f"expected int, but got {value1}"
            assert isinstance(value2, jvmc.StackInt), f"expected int, but got {value2}"
            if compare(cond,value1.value, value2.value):
                frame.pc %= targ
            else:
                frame.pc +=1
        case jvm.Goto(offset=off, target=targ):
            frame.pc %= targ
            # raise NotImplementedError(f"Got not implement  {targ} {targ!r}")
        case jvm.Store(offset=off, type= typ, index=i):
            value = frame.stack.pop()
            frame.locals[i] = value
            frame.pc +=1
        case jvm.Cast(offset=off, from_=fr, to_=to):
            v1 = frame.stack.pop()

            match to:
                case jvm.Short():
                    val_16bit = v1.value & 0xFFFF
                    result = val_16bit if val_16bit < 0x8000 else val_16bit - 0x10000
                    frame.stack.push(jvmc.StackInt(result))
                case jvm.Byte():
                    val_8bit = v1.value & 0xFF
                    result = val_8bit if val_8bit < 0x80 else val_8bit - 0x0100
                    frame.stack.push(jvmc.StackInt(result))
                case jvm.Char():
                    result = v1.value & 0xFFFF
                    frame.stack.push(jvmc.StackInt(result))
                case jvm.Int():
                    result = int(v1.value) & 0xFFFFFFFF
                    if result >= 0x80000000:
                        result -= 0x100000000
                    frame.stack.push(jvmc.StackInt(result))
                case _:
                    raise NotImplementedError(f"Cast to {to!r} ")
            frame.pc +=1
        case jvm.Get(static=True, field=field):
            # Hack - Only handle the assertion case
            assert field.extension.name == "$assertionsDisabled"

            # Hack - Assuming assertions are never disabled
            frame.stack.push(jvmc.StackInt(0))
            frame.pc += 1

        case jvm.New(classname=jvm.ClassName("java.lang.AssertionError")):
            # Hack -- if we create an assertion error, we probably also throw it.
            output = "assertion error"

        case a:
            raise NotImplementedError(a.help())

    assert isinstance(output, (jvmc.State, str))

    return pc, output


def initial(bc: jpamb.Bytecode, methodid: jvm.AbsMethodID, input: jpamb.Input):
    frame = jvmc.Frame.from_method(bc.getmethod(methodid))
    state = jvmc.State(jvmc.Heap(), jvmc.CallStack.from_frames([frame]))
    for i, v in enumerate(input.values):
        # Convert arbitrary values into local values
        match v:
            case jpamb.case.Boolean(value):
                frame.locals[i] = jvmc.StackInt(1 if value else 0)
            case jpamb.case.Int(value):
                frame.locals[i] = jvmc.StackInt(value)
            case jpamb.case.Array(contains=type, values=values):
                match type:
                    case jvm.Char():
                        ref = state.heap.new(
                            jvmc.HeapArray(type, [ord(a) for a in values])
                        )
                    case jvm.Int():
                        ref = state.heap.new(jvmc.HeapArray(type, [a for a in values]))
                frame.locals[i] = ref
            case jpamb.case.String(value=value):
                ref = state.heap.new(jvmc.HeapString(value))
                frame.locals[i] = ref
            case a:
                raise NotImplementedError(
                    f"Do not know how to convert values of type {a!r} to a local value"
                )

    return state


def interpret():
    """The entry point for the interpreter"""

    methodid, input, max_steps = jpamb.getcase(
        "dynamic",
        "1.0",
        "The Rice Theorem Cookers",
        ["dynamic", "python"],
        for_science=True,
    )

    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(suite, eff, {})

    state = initial(bc, methodid, input)

    last = jpamb.emit_init(state)

    for x in range(max_steps):
        pc, state = step(bc, state)
        last = jpamb.emit_step(last, pc, state)

        if isinstance(state, str):
            break


def fuzz_input(rand: random.Random, methodid: jvm.AbsMethodID) -> jpamb.case.Input:
    input = []
    # 1. come up with possible inputs
    for p in methodid.extension.params:
        match p:
            case jvm.Int():
                input.append(jpamb.case.Int(rand.randint(-(1 << 31), 1 << 31)))
            case jvm.Boolean():
                input.append(jpamb.case.Boolean(1 == rand.randint(0, 1)))
            case a:
                raise NotImplementedError(
                    "Don't know how to create random values for {input}"
                )

    return jpamb.case.Input(input)


def analyse():
    """The dynamic analysis, e.g. in this case a (dumb) fuzzer."""

    methodid = jpamb.getmethodid(
        "dynamic",
        "1.0",
        "The Rice Theorem Cookers",
        ["dynamic", "python"],
        for_science=True,
    )

    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(suite, eff, {})

    MAX_STEPS = 200

    import random

    # Make the randomness deterministic
    rand = random.Random(0)

    behaviors = set()
    # Try 10 random inputs
    for i in range(10):
        input = fuzz_input(rand, methodid)
        state = initial(bc, methodid, input)

        for x in range(MAX_STEPS):
            _, state = step(bc, state)
            if isinstance(state, str):
                behaviors.add(state)
                break

    for query in jpamb.QUERIES:
        if query in behaviors:
            if query == "*":
                print(f"{query};timeout")
            else:
                print(f"{query};found")
        else:
            print(f"{query};not-found")

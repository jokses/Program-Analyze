import random
import sys

import jpamb
from jpamb import case
import jvm
import jvm.state as jvmc


def binary(op, v1: int, v2: int) -> int | str:
    match op:
        case jvm.BinaryOpr.Rem:
            if v2 == 0:
                return "divide by zero"
            q = int(v1 / v2)
            return v1 - q * v2  
        case a:
            raise NotImplementedError(f"Unhandled binary {op!r}")


def compare(op, v1: int, v2: int) -> bool:
    match op:
        case jvm.CmpOpr.Gt:
            return v1 > v2
        case jvm.CmpOpr.Lt:
            return v1 < v2
        case jvm.CmpOpr.Eq:
            return v1 == v2
        case jvm.CmpOpr.Ne:
            return v1 != v2
        case jvm.CmpOpr.Ge:
            return v1 >= v2
        case jvm.CmpOpr.Le:
            return v1 <= v2
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
        case jvm.Negate(type=t):
            val = frame.stack.pop()
            frame.stack.push(jvmc.StackInt(-val.value))
            frame.pc += 1

        case jvm.Incr(index=n, amount=amt):
            v = frame.locals[n]
            frame.locals[n] = jvmc.StackInt(v.value + amt)
            frame.pc += 1
        case jvm.Push(type=t, value=v):
            match t:
                case jvm.Int():
                    frame.stack.push(jvmc.StackInt(v))
                case _:
                    frame.stack.push(None)  
            frame.pc += 1
        case jvm.Cast(from_=from_t, to_=to_t):
            val = frame.stack.pop()
            v = val.value
            if to_t == jvm.Short():
                # 16-bit signed integer
                v = v & 0xFFFF
                if v >= 0x8000:
                    v -= 0x10000
            elif to_t == jvm.Char():
                # 16-bit unsigned integer
                v = v & 0xFFFF
            elif to_t == jvm.Byte():
                # 8-bit signed integer
                v = v & 0xFF
                if v >= 0x80:
                    v -= 0x100
                    
            frame.stack.push(jvmc.StackInt(v))
            frame.pc += 1

        case jvm.Return(type=t):
            if t is not None:
                v1 = frame.stack.pop()
            state.frames.pop()
            if state.frames:
                frame = state.frames.peek()
                if t is not None:
                    frame.stack.push(v1)
                frame.pc += 1
            else:
                output = "ok"

        case jvm.Get(static=True, field=field):
            # Hack - Only handle the assertion case
            assert field.extension.name == "$assertionsDisabled"

            # Hack - Assuming assertions are never disabled
            frame.stack.push(jvmc.StackInt(0))
            frame.pc += 1
        case jvm.NewArray(type=t, dim=dim):
            assert dim == 1, "Only 1D arrays supported in this block"
            count_val = frame.stack.pop()
            count = count_val.value
            
            if count < 0:
                output = "negative array size"
            else:
                ref = state.heap.new(jvmc.HeapArray(t, [0] * count))
                frame.stack.push(ref)
                frame.pc += 1

        case jvm.New(classname=jvm.ClassName("java.lang.AssertionError")):
            # Hack -- if we create an assertion error, we probably also throw it.
            output = "assertion error"

        case jvm.Ifz(condition=op, target=target):
            value = frame.stack.pop()
            assert isinstance(value, jvmc.StackInt), f"expected int, but got {value}"
            if compare(op, value.value, 0):
                frame.pc %= target
            else:
                frame.pc += 1
        case jvm.Dup(words=w):
            if w == 1:
                v = frame.stack.pop()
                frame.stack.push(v)
                frame.stack.push(v)
                frame.pc += 1
            else:
                raise NotImplementedError(f"dup with words={w} not yet handled")
        case jvm.If(condition=op, target=target):
            v2 = frame.stack.pop()
            v1 = frame.stack.pop()
            assert isinstance(v1, jvmc.StackInt), f"expected int, but got {v1}"
            assert isinstance(v2, jvmc.StackInt), f"expected int, but got {v2}"
            if compare(op, v1.value, v2.value):
                frame.pc %= target
            else:
                frame.pc += 1
        case jvm.Binary(type=jvm.Int(), operant=op):
            v2, v1 = frame.stack.pop(), frame.stack.pop()
            assert isinstance(v1, jvmc.StackInt), f"expected int, but got {v1}"
            assert isinstance(v2, jvmc.StackInt), f"expected int, but got {v2}"

            value: int | str = binary(op, v1.value, v2.value)

            if isinstance(value, str):
                output = value
            else:
                frame.stack.push(jvmc.StackInt(value))
                frame.pc += 1
        case jvm.Load(type=t, index=n):
                    v = frame.locals[n]
                    frame.stack.push(v)
                    frame.pc += 1

        case jvm.ArrayStore(type=t):
            val = frame.stack.pop()
            idx = frame.stack.pop()
            ref = frame.stack.pop()
            
            if ref is None:
                output = "null pointer"
            else:
                arr = state.heap[ref]  # <-- Exactly like this
                if not (0 <= idx.value < len(arr.values)):
                    output = "out of bounds"
                else:
                    arr.values[idx.value] = val.value
                    frame.pc += 1
        case jvm.ArrayLoad(type=t):
            idx = frame.stack.pop()
            ref = frame.stack.pop()

            if ref is None:
                output = "null pointer"
            else:
                arr = state.heap[ref]
                if not (0 <= idx.value < len(arr.values)):
                    output = "out of bounds"
                else:
                    frame.stack.push(jvmc.StackInt(arr.values[idx.value]))
                    frame.pc += 1
        case jvm.ArrayLength():
            ref = frame.stack.pop()
            if ref is None:
                output = "null pointer"
            else:
                arr = state.heap[ref]
                frame.stack.push(jvmc.StackInt(len(arr.values)))
                frame.pc += 1

        case jvm.Goto(target=target):
            frame.pc %= target

        case jvm.Store(type=t, index=n):
            v = frame.stack.pop()
            frame.locals[n] = v
            frame.pc += 1
        
        case jvm.InvokeStatic(method=methodid):
            n = len(methodid.extension.params)
            args = [frame.stack.pop() for _ in range(n)][::-1]
            new_frame = jvmc.Frame.from_method(bc.getmethod(methodid))
            for i, v in enumerate(args):
                new_frame.locals[i] = v
            state.frames.push(new_frame)
        case a:
            raise NotImplementedError(a.help())

        

    assert isinstance(output, (jvmc.State, str))

    return pc, output

def binary(op, v1: int, v2: int) -> int | str:
    match op:
        case jvm.BinaryOpr.Add:
            return v1 + v2
        case jvm.BinaryOpr.Sub:
            return v1 - v2
        case jvm.BinaryOpr.Mul:
            return v1 * v2
        case jvm.BinaryOpr.Div:
            try:
                return v1 // v2
            except ZeroDivisionError:
                return "divide by zero"
        case jvm.BinaryOpr.Rem:
            try:
                return v1 % v2
            except ZeroDivisionError:
                return "divide by zero"
        case a:
            raise NotImplementedError(f"Unhandled binary {op!r}")

        
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
        "kindleframe",
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
    for p in methodid.extension.params:
        match p:
            case jvm.Int():
                input.append(jpamb.case.Int(rand.randint(-(1 << 31), 1 << 31)))
            case jvm.Boolean():
                input.append(jpamb.case.Boolean(1 == rand.randint(0, 1)))
            case jvm.Array(contains=t):
                length = rand.randint(0, 5)
                match t:
                    case jvm.Int():
                        vals = [rand.randint(-100, 100) for _ in range(length)]
                    case jvm.Char():
                        vals = [chr(rand.randint(97, 122)) for _ in range(length)]
                    case a:
                        raise NotImplementedError(f"no fuzz strategy for array of {a!r}")
                input.append(jpamb.case.Array(contains=t, values=vals))
            case a:
                raise NotImplementedError(
                    f"Don't know how to create random values for {a!r}"
                )

    return jpamb.case.Input(input)


def analyse():
    """The dynamic analysis, e.g. in this case a (dumb) fuzzer."""

    methodid = jpamb.getmethodid(
        "dynamic",
        "1.0",
        "kindleframe",
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


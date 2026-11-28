import bisect
import functools
import random
import string
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

from syntactic_constants import constants_for_method_id

import jpamb
import jvm
import jvm.state as jvmc

DEBUG = False

MAX_ARRAY_SIZE = 1_000_000

MAX_FUZZ_LENGTH = 8


# Java ints wrap around, like 2147483647 + 1 == -2147483648.
def to_int(value: int) -> int:
    return (value + 2**31) % 2**32 - 2**31


def binary(op, v1: int, v2: int) -> int | str:
    match op:
        case jvm.BinaryOpr.Div:
            if v2 == 0:
                return "divide by zero"
            # Java rounds toward zero and Python's // rounds down.
            quotient = abs(v1) // abs(v2)
            return to_int(quotient if (v1 < 0) == (v2 < 0) else -quotient)
        case jvm.BinaryOpr.Rem:
            if v2 == 0:
                return "divide by zero"
            # The remainder has the sign of v1, like -7 % 2 == -1 in Java.
            remainder = abs(v1) % abs(v2)
            return remainder if v1 >= 0 else -remainder
        case jvm.BinaryOpr.Sub:
            return to_int(v1 - v2)
        case jvm.BinaryOpr.Mul:
            return to_int(v1 * v2)
        case jvm.BinaryOpr.Add:
            return to_int(v1 + v2)
        case a:
            raise NotImplementedError(f"Unhandled binary {op!r}")


def bitwise(op: jvm.BitOpr, v1: int, v2: int) -> int:
    # Source: https://docs.oracle.com/javase/specs/jvms/se21/html/jvms-6.html#jvms-6.5.ishl
    # A shift only uses the low 5 bits of v2.
    shift = v2 & 0x1F
    match op:
        case jvm.BitOpr.Shl:
            return to_int(v1 << shift)
        case jvm.BitOpr.Shr:
            return v1 >> shift
        case jvm.BitOpr.Ushr:
            # Source: https://docs.oracle.com/javase/specs/jvms/se21/html/jvms-6.html#jvms-6.5.iushr
            # The bits shifted in are zeros, also for a negative v1.
            return to_int((v1 % 2**32) >> shift)
        case jvm.BitOpr.And:
            return v1 & v2
        case jvm.BitOpr.Or:
            return v1 | v2
        case jvm.BitOpr.Xor:
            return v1 ^ v2


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
    if DEBUG:
        print(f"Stepping {pc}:\n > {opr}", file=sys.stderr)
    match opr:
        case jvm.Push(type=t, value=v):
            if DEBUG:
                print(f"---value   {t}: {t!r}", file=sys.stderr)
            match t:
                case jvm.Int():
                    frame.stack.push(jvmc.StackInt(v))

                case jvm.Reference() | jvm.Array():
                    frame.stack.push(jvmc.StackReference(v))

                case jvm.Object(classname) if classname.dotted() == "java.lang.String":
                    heap_str = jvmc.HeapString(content=v)

                    str_ref = state.heap.new(heap_str)

                    frame.stack.push(str_ref)

                case _:
                    raise NotImplementedError(f"expected int, but got {jvm.Int()} {t} {t!r}")
            frame.pc += 1

        case jvm.Ifz(offset=off, condition=cond, target=targ):
            value = frame.stack.pop()
            assert isinstance(value, jvmc.StackInt), f"expected int, but got {value}"

            if compare(cond, value.value, 0):
                frame.pc = frame.pc % targ
            else:
                frame.pc += 1

        case jvm.Load(index=index):
            frame.stack.push(frame.locals[index])
            frame.pc += 1

        case jvm.InvokeStatic(offset=off, method=method_id):
            target_method = bc.getmethod(method_id)
            if DEBUG:
                print(f" Calling method: {method_id.extension.name}", file=sys.stderr)

            new_frame = jvmc.Frame.from_method(target_method)
            num_params = len(method_id.extension.params._elements)

            args = []
            for _ in range(num_params):
                args.append(frame.stack.pop())
            args.reverse()

            for i, arg_val in enumerate(args):
                new_frame.locals[i] = arg_val

            # The caller continues after the call when the new frame returns.
            frame.pc += 1
            state.frames.push(new_frame)

        case jvm.NewArray(offset=off, type=t, dim=d):
            count = frame.stack.pop().value
            if DEBUG:
                print(f" value ----  {t} {count}", file=sys.stderr)

            # JPAMB has no outcome for NegativeArraySizeException.
            if count < 0:
                return pc, "negative array size"
            # A large random size would use up the memory.
            if count > MAX_ARRAY_SIZE:
                return pc, "out of memory"
            match t:
                case jvm.Int():
                    heap_array_obj = jvm.state.HeapArray(contains=t, values=[0] * count)
                    array_ref = state.heap.new(heap_array_obj)
                    frame.stack.push(array_ref)
                    frame.pc += 1
                case _:
                    raise NotImplementedError(f"expected int, but got {jvm.Int()}  {t!r}")

        case jvm.ArrayStore(offset=off, type=t):
            value_element = frame.stack.pop()
            index_element = frame.stack.pop()
            ref_element = frame.stack.pop()

            raw_value = value_element.value
            raw_index = index_element.value

            try:
                heap_array = state.heap[ref_element]
            except IndexError:
                return pc, "null pointer"

            if not (0 <= raw_index < len(heap_array.values)):
                return pc, "out of bounds"

            heap_array.values[raw_index] = raw_value
            frame.pc += 1

        case jvm.InvokeVirtual(offset=off, method=method_id):
            classname_str = method_id.classname.dotted()
            method_name = method_id.extension.name

            if DEBUG:
                print(
                    f" Calling virtual method: {classname_str}.{method_name}",
                    file=sys.stderr,
                )

            num_params = len(method_id.extension.params._elements)

            args = []
            for _ in range(num_params):
                args.append(frame.stack.pop())
            args.reverse()

            object_ref = frame.stack.pop()

            # java.lang.String has no bytecode in the suite, so its methods run here in Python.
            if classname_str == "java.lang.String":
                heap_string_obj = state.heap[object_ref]
                raw_string = heap_string_obj.content

                match method_name:
                    case "equals":
                        other_ref = args[0]
                        other_string_obj = state.heap[other_ref]

                        is_equal = (raw_string == other_string_obj.content)
                        frame.stack.push(jvmc.StackInt(1 if is_equal else 0))

                    case "length":
                        frame.stack.push(jvmc.StackInt(len(raw_string)))

                    case "charAt":
                        idx = args[0].value
                        if not 0 <= idx < len(raw_string):
                            return pc, "out of bounds"
                        frame.stack.push(jvmc.StackInt(ord(raw_string[idx])))

                    case _:
                        raise NotImplementedError(f"The String method {method_name} is not implemented in InvokeVirtual")

                frame.pc += 1
                return pc, state

            target_method = bc.getmethod(method_id)
            new_frame = jvmc.Frame.from_method(target_method)

            new_frame.locals[0] = object_ref

            for i, arg_val in enumerate(args):
                new_frame.locals[i + 1] = arg_val

            frame.pc += 1

            state.frames.push(new_frame)

        case jvm.Negate(offset=off, type=t):
            value_element = frame.stack.pop()
            assert isinstance(value_element, jvmc.StackInt), f"expected int, but got {value_element}"

            raw_value = value_element.value
            negated_result = to_int(-raw_value)

            match t:
                case jvm.Int():
                    frame.stack.push(jvmc.StackInt(value=negated_result))
                case _:
                    raise NotImplementedError(f"Negate is not implemented for the type {t!r}")

            frame.pc += 1

        case jvm.ArrayLength(offset=off):
            ref_element = frame.stack.pop()

            try:
                heap_array = state.heap[ref_element]
            except IndexError:
                return pc, "null pointer"

            length = len(heap_array.values)
            frame.stack.push(jvm.state.StackInt(value=length))
            frame.pc += 1

        case jvm.ArrayLoad(offset=off, type=t):
            index_element = frame.stack.pop()
            ref_element = frame.stack.pop()

            raw_index = index_element.value

            try:
                heap_array = state.heap[ref_element]
            except IndexError:
                return pc, "null pointer"

            if not (0 <= raw_index < len(heap_array.values)):
                return pc, "out of bounds"

            raw_value = heap_array.values[raw_index]

            match t:
                case jvm.Int() | jvm.Char():
                    frame.stack.push(jvm.state.StackInt(value=raw_value))
                case _:
                    raise NotImplementedError(f"ArrayLoad is not implemented for the type {t!r}")

            frame.pc += 1

        case jvm.Incr(offset=off, index=idx, amount=amt):
            current_local = frame.locals[idx]

            current_value = current_local.value

            new_value = to_int(current_value + amt)

            frame.locals[idx] = jvm.state.StackInt(value=new_value)

            frame.pc += 1

        case jvm.Dup(offset=off, words=w):
            if w == 1:
                top_element = frame.stack.pop()

                frame.stack.push(top_element)
                frame.stack.push(top_element)

                frame.pc += 1
            elif w == 2:
                top = frame.stack.pop()
                below = frame.stack.pop()

                frame.stack.push(below)
                frame.stack.push(top)
                frame.stack.push(below)
                frame.stack.push(top)

                frame.pc += 1
            else:
                raise NotImplementedError(f"Dup with words={w} is not implemented")

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

        case jvm.Bitopr(type=jvm.Int(), operant=op):
            v2, v1 = frame.stack.pop(), frame.stack.pop()
            assert isinstance(v1, jvmc.StackInt), f"expected int, but got {v1}"
            assert isinstance(v2, jvmc.StackInt), f"expected int, but got {v2}"
            frame.stack.push(jvmc.StackInt(bitwise(op, v1.value, v2.value)))
            frame.pc += 1

        case jvm.Return(type=t):
            is_void = (t is None)

            if not is_void:
                v1 = frame.stack.pop()

                if len(state.frames.frames) == 1:
                    frame.stack.push(v1)
                    return pc, "ok"
                else:
                    state.frames.pop()
                    previous_frame = state.frames.peek()
                    previous_frame.stack.push(v1)
                    return pc, state
            else:
                if len(state.frames.frames) == 1:
                    return pc, "ok"
                else:
                    state.frames.pop()
                    return pc, state

        case jvm.If(offset=off, condition=cond, target=targ):
            right = frame.stack.pop()
            left = frame.stack.pop()
            assert isinstance(left, jvmc.StackInt), f"expected int, but got {left}"
            assert isinstance(right, jvmc.StackInt), f"expected int, but got {right}"

            # pc % target is the pc of the jump target.
            if compare(cond, left.value, right.value):
                frame.pc %= targ
            else:
                frame.pc += 1

        case jvm.Goto(offset=off, target=targ):
            frame.pc %= targ

        case jvm.Store(offset=off, type=typ, index=i):
            value = frame.stack.pop()
            frame.locals[i] = value
            frame.pc += 1

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
            frame.pc += 1

        case jvm.Get(static=True, field=field):
            # Hack - Only handle the assertion case
            assert field.extension.name == "$assertionsDisabled"

            # Hack - Assuming assertions are never disabled
            frame.stack.push(jvmc.StackInt(0))
            frame.pc += 1

        case jvm.InvokeSpecial(offset=off, method=method_id, is_interface=is_iface):
            num_params = len(method_id.extension.params._elements)
            for _ in range(num_params):
                frame.stack.pop()
            object_ref = frame.stack.pop()
            frame.pc += 1

        case jvm.Throw(offset=off):
            error_ref = frame.stack.pop()
            return pc, "assertion error"

        case jvm.New(offset=off) as new_opr:
            classname_obj = new_opr.classname
            dummy_error_obj = jvmc.HeapString(content=f"Exception: {classname_obj.dotted()}")
            error_ref = state.heap.new(dummy_error_obj)
            frame.stack.push(error_ref)
            frame.pc += 1

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
        "JK_Interpret",
        "1.0",
        "Joachim",
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
                input.append(jpamb.case.Int(rand.randint(-(1 << 31), (1 << 31) - 1)))
            case jvm.Boolean():
                input.append(jpamb.case.Boolean(1 == rand.randint(0, 1)))
            # Source: https://courses.compute.dtu.dk/02242/topics/dynamic-analysis.html#sec:4.1
            case jvm.Array(contains=jvm.Int()):
                length = rand.randint(0, MAX_FUZZ_LENGTH)
                values = [
                    rand.randint(-(1 << 31), (1 << 31) - 1) for _ in range(length)
                ]
                input.append(jpamb.case.Array(jvm.Int(), tuple(values)))
            case jvm.Array(contains=jvm.Char()):
                length = rand.randint(0, MAX_FUZZ_LENGTH)
                values = rand.choices(string.ascii_lowercase, k=length)
                input.append(jpamb.case.Array(jvm.Char(), tuple(values)))
            case jvm.Object(classname) if classname.dotted() == "java.lang.String":
                length = rand.randint(0, MAX_FUZZ_LENGTH)
                letters = rand.choices(string.ascii_lowercase, k=length)
                input.append(jpamb.case.String(value="".join(letters)))


            case a:
                raise NotImplementedError(
                    f"Don't know how to create random values for {p}"
                )

    return jpamb.case.Input(input)


# Source: https://github.com/kalhauge/jpamb/commit/7a7a36af695aae8638be4983e291d456d4eff277
@dataclass
class Draw:
    code: bytes
    offset: int = 0

    def next_byte(self) -> int:
        value = self.code[self.offset] if self.offset < len(self.code) else 0
        self.offset += 1
        return value

    def next_32bit_signed_int(self) -> int:
        value = int.from_bytes(self.code[self.offset : self.offset + 4], signed=True)
        self.offset += 4
        return value


def fuzz_type(draw: Draw, target: jvm.Type) -> jpamb.case.Value:
    match target:
        case jvm.Int():
            return jpamb.case.Int(draw.next_32bit_signed_int())
        case jvm.Boolean():
            return jpamb.case.Boolean(draw.next_byte() % 2 == 1)
        case jvm.Array(contains=jvm.Int()):
            length = draw.next_byte() % (MAX_FUZZ_LENGTH + 1)
            values = tuple(draw.next_32bit_signed_int() for _ in range(length))
            return jpamb.case.Array(jvm.Int(), values)
        case jvm.Array(contains=jvm.Char()):
            length = draw.next_byte() % (MAX_FUZZ_LENGTH + 1)
            values = tuple(chr(draw.next_byte()) for _ in range(length))
            return jpamb.case.Array(jvm.Char(), values)
        case jvm.Object(classname) if classname.dotted() == "java.lang.String":
            length = draw.next_byte() % (MAX_FUZZ_LENGTH + 1)
            return jpamb.case.String(
                "".join(chr(draw.next_byte()) for _ in range(length))
            )
        case a:
            raise NotImplementedError(f"{a!r}")


# Source: https://courses.compute.dtu.dk/02242/topics/dynamic-analysis.html#sec:4.2
def ints_in_method(method: jvm.Method) -> set[int]:
    ints = set()
    for opcode in method.opcodes:
        match opcode:
            case jvm.Push(type=jvm.Int(), value=value):
                ints.add(value)
            case jvm.Incr(amount=amount):
                ints.add(amount)
    return ints


def strings_in_method(method: jvm.Method) -> list[str]:
    return [
        opcode.value
        for opcode in method.opcodes
        if isinstance(opcode, jvm.Push) and isinstance(opcode.value, str)
    ]


# fuzz_type reads these bytes as the constant in every parameter.
def constant_code(params: list[jvm.Type], constant: int | str) -> bytes:
    number = constant if isinstance(constant, int) else 0
    if isinstance(constant, str):
        text = bytes(ord(char) % 256 for char in constant[:MAX_FUZZ_LENGTH])
    else:
        text = bytes([constant]) if 0 <= constant < 256 else b""
    code = b""
    for param in params:
        match param:
            case jvm.Int():
                code += number.to_bytes(4, signed=True)
            case jvm.Boolean():
                code += bytes([1 if number else 0])
            case jvm.Array(contains=jvm.Int()):
                code += bytes([1]) + number.to_bytes(4, signed=True)
            case _:
                code += bytes([len(text)]) + text
    return code


@dataclass(frozen=True)
class Trace:
    outcome: str
    instructions: frozenset[jvmc.PC]
    # (pc, operand, low, high) means the operand at pc was in [low, high].
    ranges: frozenset[tuple[jvmc.PC, int, int, int]] = frozenset()
    # The executed instructions in order.
    path: tuple[jvmc.PC, ...] = ()


# A site is split at the comparisons in its own method. The cache is there because
# syntactic_constants reads the Java source every time.
@functools.cache
def comparisons_in_method(methodid: jvm.AbsMethodID) -> frozenset[int]:
    return constants_for_method_id(methodid.encode(), follow_calls=False).comparisons


# arraySumIsLarge compares the sum with 300. With "three" its points are 0 and 300 and
# the ranges are [-2**31, -1], [0, 0], [1, 299], [300, 300] and [301, 2**31 - 1].
# For a constant c, "three" uses c as a point and "boundary" uses c-1, c and c+1.
def split_points(constants: set[int], split: str) -> list[int]:
    # Source: https://courses.compute.dtu.dk/02242/topics/unbounded-static-analysis.html#sec:2.3
    points = set()
    for constant in constants | {0}:
        match split:
            case "three":
                points.add(constant)
            case "boundary":
                points.update({constant - 1, constant, constant + 1})
            case a:
                raise NotImplementedError(f"{a!r}")
    return sorted(point for point in points if -(2**31) <= point < 2**31)


def range_of(value: int, points: list[int]) -> tuple[int, int]:
    index = bisect.bisect_left(points, value)
    if index < len(points) and points[index] == value:
        return (value, value)
    low = points[index - 1] + 1 if index > 0 else -(2**31)
    high = points[index] - 1 if index < len(points) else 2**31 - 1
    return (low, high)


# The operands we record a range for, from the stack values with the top last.
def operands(opcode: jvm.Opcode, values: Sequence) -> list:
    match opcode:
        case jvm.If() | jvm.Binary(operant=jvm.BinaryOpr.Div | jvm.BinaryOpr.Rem):
            return [values[-2], values[-1]]
        case jvm.Ifz():
            return [values[-1]]
        case jvm.ArrayLoad():
            # The stack is array, index.
            return [values[-1]]
        case jvm.ArrayStore():
            # The stack is array, index, value.
            return [values[-2]]
        case _:
            return []


def run(
    bc: jpamb.Bytecode,
    methodid: jvm.AbsMethodID,
    input: jpamb.case.Input,
    *,
    max_steps: int,
    split: str | None = None,
) -> Trace:
    state = initial(bc, methodid, input)
    instructions = set()
    ranges = set()
    path = []
    points = {}
    for _ in range(max_steps):
        frame = state.frames.peek()
        if split is not None:
            method = frame.pc.method
            if method not in points:
                points[method] = split_points(set(comparisons_in_method(method)), split)
            # The operands are read before step pops them.
            stack = frame.stack.operands
            for position, operand in enumerate(operands(bc[frame.pc], stack)):
                # Reference comparisons like if_acmpeq have no int range.
                if isinstance(operand, jvmc.StackInt):
                    low, high = range_of(operand.value, points[method])
                    ranges.add((frame.pc, position, low, high))
        pc, state = step(bc, state)
        instructions.add(pc)
        path.append(pc)
        if isinstance(state, str):
            return Trace(state, frozenset(instructions), frozenset(ranges), tuple(path))
    return Trace("*", frozenset(instructions), frozenset(ranges), tuple(path))


# Source: https://courses.compute.dtu.dk/02242/topics/dynamic-analysis.html#sec:4.3
@dataclass
class Corpus:
    params: list[jvm.Type]
    rand: random.Random
    interesting: list[bytes] = field(default_factory=list)
    coverage: set = field(default_factory=set)

    def fuzz_input(self, code: bytes) -> jpamb.case.Input:
        draw = Draw(code)
        return jpamb.case.Input(tuple(fuzz_type(draw, param) for param in self.params))

    def next(self) -> bytes:
        code = bytearray(self.rand.choice(self.interesting))
        match self.rand.randrange(3):
            case 0 if code:
                code[self.rand.randrange(len(code))] = self.rand.randrange(256)
            case 1 if code:
                del code[self.rand.randrange(len(code))]
            case _:
                code.append(self.rand.randrange(256))
        return bytes(code)

    def add(self, code: bytes, coverage: set) -> bool:
        if coverage <= self.coverage:
            return False
        self.coverage |= coverage
        self.interesting.append(code)
        return True


def analyse():
    """The dynamic analysis, e.g. in this case a (dumb) fuzzer."""
    methodid = jpamb.getmethodid(
        "JK_Analyse",
        "1.0",
        "Joachim",
        ["dynamic", "python"],
        for_science=True
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


import sys
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from abstractions import SignSet

import jpamb
import sexpr
from jpamb import jvm
import jvm.state as jvms
from jvm.state import PC, StackInt


# ============================================================
# Abstract values
# ============================================================

MAX_CONSTANTS = 32


@dataclass(frozen=True)
class IntAbs(sexpr.AsSExpr):
    """
    Integer abstraction.

    If values is not None, we know a finite set of concrete integers.
    If values is None, we use the sign abstraction.
    """

    values: frozenset[int] | None = None
    signs: SignSet = field(default_factory=SignSet.top)

    @classmethod
    def concrete(cls, value: int) -> "IntAbs":
        return cls(
            values=frozenset({value}),
            signs=SignSet.abstract([StackInt(value)]),
        )

    @classmethod
    def from_values(cls, values: Iterable[int]) -> "IntAbs":
        values = frozenset(values)

        if not values:
            return cls(frozenset(), SignSet.bot())

        if len(values) <= MAX_CONSTANTS:
            return cls(
                values=values,
                signs=SignSet.abstract(
                    StackInt(v) for v in values
                ),
            )

        return cls(
            values=None,
            signs=SignSet.abstract(
                StackInt(v) for v in values
            ),
        )

    @classmethod
    def top(cls) -> "IntAbs":
        return cls(None, SignSet.top())

    @classmethod
    def bot(cls) -> "IntAbs":
        return cls(frozenset(), SignSet.bot())

    def __or__(self, other: "IntAbs") -> "IntAbs":
        if self.values is not None and other.values is not None:
            values = self.values | other.values

            if len(values) <= MAX_CONSTANTS:
                return IntAbs.from_values(values)

            return IntAbs(
                None,
                self.signs | other.signs,
            )

        return IntAbs(
            None,
            self.signs | other.signs,
        )

    def is_bot(self) -> bool:
        return self.values == frozenset()

    def is_concrete(self) -> bool:
        return self.values is not None and len(self.values) == 1

    def concrete_value(self) -> int | None:
        if self.is_concrete():
            return next(iter(self.values))
        return None
    
    def possible_zero(self) -> bool:
        if self.values is not None:
            return 0 in self.values

        return 0 in self.signs.signs

    def possible_nonzero(self) -> bool:
        if self.values is not None:
            return any(v != 0 for v in self.values)

        return any(v != 0 for v in self.signs.signs)

    def concrete_values(self) -> set[int] | None:
        if self.values is None:
            return None
        return set(self.values)

    def arithmetic(
        self,
        other: "IntAbs",
        op: jvm.BinaryOpr,
    ) -> tuple["IntAbs", set[str]]:
        errors: set[str] = set()

        def jvm_div(x: int, y: int) -> int:
            # Java/JVM integer division truncates toward zero.
            q = abs(x) // abs(y)
            return -q if (x < 0) != (y < 0) else q

        def jvm_rem(x: int, y: int) -> int:
            # Java remainder: x - trunc(x/y) * y
            return x - jvm_div(x, y) * y

        if self.values is not None and other.values is not None:
            output: set[int] = set()

            for x in self.values:
                for y in other.values:
                    match op:
                        case jvm.BinaryOpr.Add:
                            output.add(x + y)

                        case jvm.BinaryOpr.Sub:
                            output.add(x - y)

                        case jvm.BinaryOpr.Mul:
                            output.add(x * y)

                        case jvm.BinaryOpr.Div:
                            if y == 0:
                                errors.add("divide by zero")
                            else:
                                output.add(jvm_div(x, y))

                        case jvm.BinaryOpr.Rem:
                            if y == 0:
                                errors.add("divide by zero")
                            else:
                                output.add(jvm_rem(x, y))

                        case _:
                            raise NotImplementedError(
                                f"Unhandled binary operation {op!r}"
                            )

            if output and len(output) <= MAX_CONSTANTS:
                return IntAbs.from_values(output), errors

            if not output:
                return IntAbs.bot(), errors

        signs, sign_errors = self.signs.arithmetic(
            other.signs,
            op,
        )

        errors |= sign_errors

        return IntAbs(None, signs), errors

    def join(self, other: "IntAbs") -> "IntAbs":
        if self == other:
            return self
        
        if self.values is not None and other.values is not None:
            combined = self.values | other.values
            if len(combined) <= MAX_CONSTANTS:
                return IntAbs.from_values(combined)
            return IntAbs(None, self.signs | other.signs)

        return IntAbs(None, self.signs | other.signs)

    @staticmethod
    def compare(
        left: "IntAbs",
        right: "IntAbs",
        op: jvm.CmpOpr,
    ) -> set[bool]:

        if left.values is not None and right.values is not None:
            result: set[bool] = set()

            for x in left.values:
                for y in right.values:
                    match op:
                        case jvm.CmpOpr.Eq:
                            result.add(x == y)
                        case jvm.CmpOpr.Ne:
                            result.add(x != y)
                        case jvm.CmpOpr.Lt:
                            result.add(x < y)
                        case jvm.CmpOpr.Le:
                            result.add(x <= y)
                        case jvm.CmpOpr.Gt:
                            result.add(x > y)
                        case jvm.CmpOpr.Ge:
                            result.add(x >= y)
                        case _:
                            raise NotImplementedError(
                                f"Unhandled comparison {op!r}"
                            )

            return result

        return SignSet.compare(
            left.signs,
            right.signs,
            op,
        )

    def __str__(self):
        if self.values is not None:
            if len(self.values) == 0:
                return "bot"
            if len(self.values) == 1:
                return str(next(iter(self.values)))
            return "{" + ",".join(
                str(v) for v in sorted(self.values)
            ) + "}"

        return str(self.signs)


@dataclass(frozen=True)
class RefAbs:
    """
    Abstract reference.

    null=True means null is possible.
    tokens identifies abstract heap objects.
    """

    tokens: frozenset[str] = frozenset()
    null: bool = False

    @classmethod
    def null_ref(cls) -> "RefAbs":
        return cls(frozenset(), True)

    @classmethod
    def unknown(cls, token: str = "unknown") -> "RefAbs":
        return cls(frozenset({token}), True)

    def __or__(self, other: "RefAbs") -> "RefAbs":
        return RefAbs(
            self.tokens | other.tokens,
            self.null or other.null,
        )


@dataclass(frozen=True)
class StringAbs:
    values: frozenset[str] | None

    @classmethod
    def concrete(cls, value: str) -> "StringAbs":
        return cls(frozenset({value}))

    @classmethod
    def top(cls) -> "StringAbs":
        return cls(None)

    def __or__(self, other: "StringAbs") -> "StringAbs":
        if self.values is None or other.values is None:
            return StringAbs.top()

        values = self.values | other.values

        if len(values) > MAX_CONSTANTS:
            return StringAbs.top()

        return StringAbs(values)


@dataclass(frozen=True)
class ArrayAbs:
    element_type: object
    length: IntAbs
    elements: tuple[IntAbs, ...] | None
    nullable: bool = False

    @classmethod
    def concrete(
        cls,
        element_type,
        values: list[int],
    ) -> "ArrayAbs":
        return cls(
            element_type=element_type,
            length=IntAbs.concrete(len(values)),
            elements=tuple(
                IntAbs.concrete(value)
                for value in values
            ),
            nullable=False,
        )

    @classmethod
    def unknown(
        cls,
        element_type,
    ) -> "ArrayAbs":
        return cls(
            element_type=element_type,
            length=IntAbs.top(),
            elements=None,
            nullable=False,
        )

    def __or__(self, other: "ArrayAbs") -> "ArrayAbs":
        if self.elements is not None and other.elements is not None:
            if len(self.elements) == len(other.elements):
                elements = tuple(
                    x | y
                    for x, y in zip(
                        self.elements,
                        other.elements,
                    )
                )
            else:
                elements = None
        else:
            elements = None

        return ArrayAbs(
            element_type=self.element_type,
            length=self.length | other.length,
            elements=elements,
            nullable=self.nullable or other.nullable,
        )


AbstractObject = ArrayAbs | StringAbs


# ============================================================
# Call frames
# ============================================================

@dataclass(frozen=True)
class CallFrame:
    return_pc: PC
    locals: tuple[object, ...]
    stack: tuple[object, ...]


# ============================================================
# Abstract state
# ============================================================

@dataclass(frozen=True)
class State(sexpr.AsSExpr):
    locals: tuple[object, ...]
    stack: tuple[object, ...]
    heap: tuple[tuple[str, AbstractObject], ...] = ()
    calls: tuple[CallFrame, ...] = ()

    def __post_init__(self):
        assert isinstance(self.locals, tuple)
        assert isinstance(self.stack, tuple)
        assert isinstance(self.heap, tuple)
        assert isinstance(self.calls, tuple)

    def __str__(self):
        return (
            f"{', '.join(map(str, self.locals))}/"
            f"{':'.join(map(str, self.stack))}"
        )

    def __sexpr__(self):
        return str(self)

    def heap_dict(self) -> dict[str, AbstractObject]:
        return dict(self.heap)

    def with_heap(
        self,
        heap: dict[str, AbstractObject],
    ) -> "State":
        return State(
            self.locals,
            self.stack,
            tuple(sorted(heap.items(), key=lambda x: x[0])),
            self.calls,
        )

    def __or__(self, other):
        assert isinstance(other, State)

        if len(self.stack) != len(other.stack):
            return self

        if len(self.locals) != len(other.locals):
            return self

        locals_joined = tuple(
            join_value(a, b)
            for a, b in zip(
                self.locals,
                other.locals,
            )
        )

        stack_joined = tuple(
            join_value(a, b)
            for a, b in zip(
                self.stack,
                other.stack,
            )
        )

        heap1 = self.heap_dict()
        heap2 = other.heap_dict()

        heap = {}

        for key in heap1.keys() | heap2.keys():
            if key in heap1 and key in heap2:
                heap[key] = join_value(
                    heap1[key],
                    heap2[key],
                )
            elif key in heap1:
                heap[key] = heap1[key]
            else:
                heap[key] = heap2[key]

        # Mantener los marcos de llamada del estado actual
        calls = self.calls if len(self.calls) >= len(other.calls) else other.calls

        return State(
            locals_joined,
            stack_joined,
            tuple(sorted(heap.items(), key=lambda x: x[0])),
            calls,
        )

    def push(self, value):
        return State(
            self.locals,
            self.stack + (value,),
            self.heap,
            self.calls,
        )

    def pop(self, number=1):
        return (
            self.stack[-number:],
            State(
                self.locals,
                self.stack[:-number],
                self.heap,
                self.calls,
            ),
        )

    def load(self, index):
        return self.locals[index]

    def store(self, index, value):
        return State(
            tuple(
                self.locals[:index]
            )
            + (value,)
            + tuple(self.locals[index + 1:]),
            self.stack,
            self.heap,
            self.calls,
        )


def join_value(a, b):
    if hasattr(a, "join") and type(a) is type(b):
        return a.join(b)

    if isinstance(a, IntAbs) and isinstance(b, IntAbs):
        return a.join(b)

    if isinstance(a, RefAbs) and isinstance(b, RefAbs):
        return a | b

    if isinstance(a, StringAbs) and isinstance(b, StringAbs):
        return a | b

    if isinstance(a, ArrayAbs) and isinstance(b, ArrayAbs):
        return a | b

    return a


# ============================================================
# Helpers
# ============================================================

def int_value(value) -> IntAbs:
    if isinstance(value, IntAbs):
        return value

    raise TypeError(f"Expected IntAbs, got {value!r}")


def ref_value(value) -> RefAbs:
    if isinstance(value, RefAbs):
        return value

    raise TypeError(f"Expected RefAbs, got {value!r}")


def object_for_ref(
    state: State,
    ref: RefAbs,
) -> list[AbstractObject]:
    heap = state.heap_dict()

    return [
        heap[token]
        for token in ref.tokens
        if token in heap
    ]


def possible_null(ref: RefAbs) -> bool:
    return ref.null


def possible_nonnull(ref: RefAbs) -> bool:
    return bool(ref.tokens)


def array_element(
    array: ArrayAbs,
    index: IntAbs,
) -> tuple[IntAbs, set[str]]:

    errors: set[str] = set()

    if index.values is not None:
        if array.length.values is not None:
            length_values = array.length.values

            for i in index.values:
                if not any(
                    0 <= i < length
                    for length in length_values
                ):
                    errors.add("out of bounds")

            valid_indices = [
                i
                for i in index.values
                if any(
                    0 <= i < length
                    for length in length_values
                )
            ]

            if (
                array.elements is not None
                and len(length_values) == 1
            ):
                result = IntAbs.bot()

                for i in valid_indices:
                    if 0 <= i < len(array.elements):
                        result = result | array.elements[i]

                return result, errors

        if array.elements is not None:
            result = IntAbs.bot()

            for i in index.values:
                if 0 <= i < len(array.elements):
                    result = result | array.elements[i]
                else:
                    errors.add("out of bounds")

            return result, errors

    # Unknown index/length.
    if array.elements is not None:
        result = IntAbs.bot()

        for value in array.elements:
            result = result | value

        return result, errors

    return IntAbs.top(), errors


def refine_int_by_comparison(
    value: IntAbs,
    op: jvm.CmpOpr,
    result: bool,
    other: IntAbs,
) -> IntAbs:
    """
    Refine an abstract integer according to:

        value <op> other == result

    We first use concrete values when available.
    Otherwise we refine the sign abstraction when the
    comparison is against a concrete zero.
    """

    # ------------------------------------------------------------
    # Precise finite-value refinement
    # ------------------------------------------------------------

    if (
        value.values is not None
        and other.values is not None
    ):
        valid = set()

        for x in value.values:
            for y in other.values:
                condition = (
                    (op == jvm.CmpOpr.Eq and (x == y) == result)
                    or
                    (op == jvm.CmpOpr.Ne and (x != y) == result)
                    or
                    (op == jvm.CmpOpr.Lt and (x < y) == result)
                    or
                    (op == jvm.CmpOpr.Le and (x <= y) == result)
                    or
                    (op == jvm.CmpOpr.Gt and (x > y) == result)
                    or
                    (op == jvm.CmpOpr.Ge and (x >= y) == result)
                )

                if condition:
                    valid.add(x)

        return IntAbs.from_values(valid)

    # ------------------------------------------------------------
    # Sign refinement for comparisons against zero
    # ------------------------------------------------------------

    if other.is_concrete() and other.concrete_value() == 0:
        allowed = set(value.signs.signs)

        if op == jvm.CmpOpr.Eq:
            if result:
                allowed &= {0}
            else:
                allowed -= {0}

        elif op == jvm.CmpOpr.Ne:
            if result:
                allowed -= {0}
            else:
                allowed &= {0}

        elif op == jvm.CmpOpr.Lt:
            if result:
                allowed &= {-1}
            else:
                allowed &= {0, 1}

        elif op == jvm.CmpOpr.Le:
            if result:
                allowed &= {-1, 0}
            else:
                allowed &= {1}

        elif op == jvm.CmpOpr.Gt:
            if result:
                allowed &= {1}
            else:
                allowed &= {-1, 0}

        elif op == jvm.CmpOpr.Ge:
            if result:
                allowed &= {0, 1}
            else:
                allowed &= {-1}

        return IntAbs(
            None,
            SignSet(frozenset(allowed)),
        )

    return value



# ============================================================
# Abstract transition relation
# ============================================================

def manystep(
    bc: jpamb.Bytecode,
    pc: PC,
    state: State,
) -> Iterable[tuple[PC, object] | str]:

    opr = bc[pc]

    # --------------------------------------------------------
    # Push
    # --------------------------------------------------------

    match opr:

        case jvm.Push(type=t, value=v):

            if t == jvm.Int():
                yield (
                    pc + 1,
                    state.push(IntAbs.concrete(v)),
                )
                return

            if isinstance(t, jvm.Reference):
                if v is None:
                    value = RefAbs.null_ref()
                else:
                    value = RefAbs.unknown(str(v))

                yield pc + 1, state.push(value)
                return

            if str(t) == "Ljava/lang/String;":
                token = f"string:{pc}"
                heap = state.heap_dict()
                heap[token] = StringAbs.concrete(v)

                new_state = state.with_heap(heap)
                yield (
                    pc + 1,
                    new_state.push(
                        RefAbs(frozenset({token}), False)
                    ),
                )
                return

        # ----------------------------------------------------
        # Assertions
        # ----------------------------------------------------

        case jvm.Get(static=True, field=field):

            assert field.extension.name == "$assertionsDisabled"

            yield (
                pc + 1,
                state.push(IntAbs.concrete(0)),
            )
            return

        case jvm.New(
            classname=jvm.ClassName("java.lang.AssertionError")
        ):
            yield "assertion error"
            return

        # ----------------------------------------------------
        # Load/store
        # ----------------------------------------------------

        case jvm.Load(index=i):
            yield pc + 1, state.push(state.load(i))
            return

        case jvm.Load(type=jvm.Reference(), index=i):
            yield pc + 1, state.push(state.load(i))
            return

        case jvm.Store(index=i):
            [value], after = state.pop()

            yield pc + 1, after.store(i, value)
            return

        case jvm.Store(type=jvm.Int(), index=i):
            [value], after = state.pop()

            yield pc + 1, after.store(i, value)
            return

        case jvm.Store(type=jvm.Reference(), index=i):
            [value], after = state.pop()

            yield pc + 1, after.store(i, value)
            return

        # ----------------------------------------------------
        # Increment
        # ----------------------------------------------------

        case jvm.Incr(index=i, amount=amount):

            value = int_value(state.load(i))

            result, errors = value.arithmetic(
                IntAbs.concrete(amount),
                jvm.BinaryOpr.Add,
            )

            for error in errors:
                yield error

            if not result.is_bot():
                yield (
                    pc + 1,
                    state.store(i, result),
                )

            return

        # ----------------------------------------------------
        # Binary arithmetic
        # ----------------------------------------------------

        case jvm.Binary(operant=op):

            [left, right], after = state.pop(2)

            left = int_value(left)
            right = int_value(right)

            result, errors = left.arithmetic(right, op)

            for error in errors:
                yield error

            if not result.is_bot():
                yield pc + 1, after.push(result)

            return

        # ----------------------------------------------------
        # If zero
        # ----------------------------------------------------

        case jvm.Ifz(condition=op, target=target):
            [value], after = state.pop()

            if isinstance(value, IntAbs):
                zero = IntAbs.concrete(0)

                results = IntAbs.compare(
                    value,
                    zero,
                    op,
                )

                for result in results:
                    refined = refine_int_by_comparison(
                        value,
                        op,
                        result,
                        zero,
                    )

                    if refined.is_bot():
                        continue

                    branch_state = after

                    # Refine locals that contain this abstract value.
                    for i, local in enumerate(after.locals):
                        if local == value and isinstance(local, IntAbs):
                            branch_state = branch_state.store(
                                i,
                                refined,
                            )

                    if result:
                        yield pc % target, branch_state
                    else:
                        yield pc + 1, branch_state

                return

            if isinstance(value, RefAbs):
                if op == jvm.CmpOpr.Eq:
                    results = {value.null}
                elif op == jvm.CmpOpr.Ne:
                    results = {not value.null}
                else:
                    raise NotImplementedError(
                        f"Invalid reference Ifz condition: {op!r}"
                    )

                for result in results:
                    if result:
                        yield pc % target, after
                    else:
                        yield pc + 1, after

                return

            raise TypeError(
                f"Unsupported Ifz value: {value!r}"
            )

        # ----------------------------------------------------
        # If two values
        # ----------------------------------------------------

        case jvm.If(condition=op, target=target):

            [left, right], after = state.pop(2)

            left = int_value(left)
            right = int_value(right)

            for result in IntAbs.compare(left, right, op):
                refined_left = left
                refined_right = right

                if left.values is not None and right.values is not None:
                    valid_left = set()
                    valid_right = set()

                    for x in left.values:
                        for y in right.values:
                            cond_holds = (
                                (op == jvm.CmpOpr.Eq and (x == y) == result)
                                or (op == jvm.CmpOpr.Ne and (x != y) == result)
                                or (op == jvm.CmpOpr.Lt and (x < y) == result)
                                or (op == jvm.CmpOpr.Le and (x <= y) == result)
                                or (op == jvm.CmpOpr.Gt and (x > y) == result)
                                or (op == jvm.CmpOpr.Ge and (x >= y) == result)
                            )
                            if cond_holds:
                                valid_left.add(x)
                                valid_right.add(y)

                    # SOLO refinar si encontramos valores válidos; 
                    # de lo contrario conservamos el valor abstracto previo para no cortar la cobertura.
                    if valid_left:
                        refined_left = IntAbs.from_values(valid_left)
                    if valid_right:
                        refined_right = IntAbs.from_values(valid_right)

                branch_state = after
                for i, local in enumerate(after.locals):
                    if local == left and isinstance(local, IntAbs):
                        branch_state = branch_state.store(i, refined_left)
                    elif local == right and isinstance(local, IntAbs):
                        branch_state = branch_state.store(i, refined_right)

                if result:
                    yield pc % target, branch_state
                else:
                    yield pc + 1, branch_state

            return


        # ----------------------------------------------------
        # Goto
        # ----------------------------------------------------

        case jvm.Goto(target=target):
            yield pc % target, state
            return

        # ----------------------------------------------------
        # New array
        # ----------------------------------------------------

        case jvm.NewArray(type=element_type, dim=1):

            [length], after = state.pop()
            length = int_value(length)

            if length.values is not None and len(length.values) == 1:
                n = next(iter(length.values))

                if n < 0:
                    yield "negative array size"
                    return

                if n <= 64:
                    array = ArrayAbs.concrete(
                        element_type,
                        [0] * n,
                    )
                else:
                    array = ArrayAbs.unknown(element_type)

            else:
                array = ArrayAbs.unknown(element_type)

            token = f"array:{pc}"

            heap = after.heap_dict()
            heap[token] = array

            after = after.with_heap(heap)

            yield (
                pc + 1,
                after.push(
                    RefAbs(
                        frozenset({token}),
                        False,
                    )
                ),
            )
            return

        # ----------------------------------------------------
        # Array length
        # ----------------------------------------------------

        case jvm.ArrayLength():

            [ref], after = state.pop()
            ref = ref_value(ref)

            if possible_null(ref):
                yield "null pointer"

            for array in object_for_ref(state, ref):
                if isinstance(array, ArrayAbs):
                    yield (
                        pc + 1,
                        after.push(array.length),
                    )

            return

        # ----------------------------------------------------
        # Array load
        # ----------------------------------------------------
        case jvm.ArrayLoad(type=_):
            [ref, index], after = state.pop(2)

            ref = ref_value(ref)
            index = int_value(index)

            if possible_null(ref):
                yield "null pointer"

            for array in object_for_ref(state, ref):
                if not isinstance(array, ArrayAbs):
                    continue

                # 1. Comprobar si puede ocurrir out of bounds
                can_be_oob = False
                can_be_in_bounds = False

                if index.values is not None and array.length.values is not None:
                    for i in index.values:
                        for l in array.length.values:
                            if i < 0 or i >= l:
                                can_be_oob = True
                            else:
                                can_be_in_bounds = True
                else:
                    # En abstracción completa (None), ambos casos son posibles
                    can_be_oob = True
                    can_be_in_bounds = True

                if can_be_oob:
                    yield "out of bounds"

                if can_be_in_bounds:
                    value, errors = array_element(array, index)
                    for error in errors:
                        yield error

                    if not value.is_bot():
                        yield (pc + 1, after.push(value))

            return
        # ----------------------------------------------------
        # Array store
        # ----------------------------------------------------
        case jvm.ArrayStore(type=_):
            [ref, index, value], after = state.pop(3)

            ref = ref_value(ref)
            index = int_value(index)
            value = int_value(value)

            if possible_null(ref):
                yield "null pointer"

            heap = state.heap_dict()

            for token in ref.tokens:
                array = heap.get(token)

                if not isinstance(array, ArrayAbs):
                    continue

                # Comprobar posibles desbordamientos
                can_be_oob = False
                can_be_in_bounds = False

                if index.values is not None and array.length.values is not None:
                    for i in index.values:
                        for l in array.length.values:
                            if i < 0 or i >= l:
                                can_be_oob = True
                            else:
                                can_be_in_bounds = True
                else:
                    can_be_oob = True
                    can_be_in_bounds = True

                if can_be_oob:
                    yield "out of bounds"

                if not can_be_in_bounds:
                    continue

                # Actualización de arreglo si el acceso dentro de límites es posible
                if index.is_concrete() and array.elements is not None:
                    i = index.concrete_value()
                    assert i is not None

                    if 0 <= i < len(array.elements):
                        elements = list(array.elements)
                        elements[i] = value
                        new_array = ArrayAbs(
                            element_type=array.element_type,
                            length=array.length,
                            elements=tuple(elements),
                            nullable=array.nullable,
                        )
                    else:
                        continue
                else:
                    # Actualización débil (weak update)
                    if array.elements is not None:
                        elements = tuple(element | value for element in array.elements)
                    else:
                        elements = None

                    new_array = ArrayAbs(
                        element_type=array.element_type,
                        length=array.length,
                        elements=elements,
                        nullable=array.nullable,
                    )

                new_heap = dict(heap)
                new_heap[token] = new_array
                new_state = after.with_heap(new_heap)

                yield (pc + 1, new_state)

            return
        # ----------------------------------------------------
        # Static call
        # ----------------------------------------------------
        case jvm.InvokeStatic(method=method_id):
            method = bc.getmethod(method_id)
            num_args = len(method_id.extension.params)

            args, after = state.pop(num_args)

            new_locals = [IntAbs.top()] * method.max_locals

            # Asignar los argumentos recibidos a las variables locales del callee
            for i, value in enumerate(args):
                new_locals[i] = value

            call = CallFrame(
                return_pc=pc + 1,
                locals=after.locals,
                stack=after.stack,
            )

            callee_state = State(
                tuple(new_locals),
                (),
                after.heap,
                after.calls + (call,),
            )

            yield PC(method_id, 0), callee_state
            return


        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------
        case jvm.Return(type=None):
            if not state.calls:
                yield "ok"
                return

            caller = state.calls[-1]

            yield (
                caller.return_pc,
                State(
                    caller.locals,
                    caller.stack,
                    state.heap,
                    state.calls[:-1],
                ),
            )
            return

        case jvm.Return(type=_):
            [value], after = state.pop()

            if not state.calls:
                yield "ok"
                return

            caller = state.calls[-1]

            # Reconstruir el estado del llamante colocando el valor en su stack
            caller_state = State(
                caller.locals,
                caller.stack + (value,),
                state.heap,
                state.calls[:-1],
            )

            yield (
                caller.return_pc,
                caller_state,
            )
            return
        
        
        # ----------------------------------------------------
        # String.equals
        # ----------------------------------------------------

        case jvm.InvokeVirtual(method=method_id):

            if (
                method_id.classname
                == jvm.ClassName("java.lang.String")
                and method_id.extension.name == "equals"
            ):
                [other, this], after = state.pop(2)

                this = ref_value(this)
                other = ref_value(other)

                this_objects = object_for_ref(state, this)
                other_objects = object_for_ref(state, other)

                results: set[bool] = set()

                for a in this_objects:
                    for b in other_objects:
                        if not isinstance(a, StringAbs):
                            continue
                        if not isinstance(b, StringAbs):
                            continue

                        if a.values is None or b.values is None:
                            results.update({True, False})
                        else:
                            for x in a.values:
                                for y in b.values:
                                    results.add(x == y)

                if not results:
                    results = {True, False}

                for result in results:
                    yield (
                        pc + 1,
                        after.push(
                            IntAbs.concrete(
                                1 if result else 0
                            )
                        ),
                    )

                return

            raise NotImplementedError(
                f"Unsupported virtual method: {method_id}"
            )

        # ----------------------------------------------------
        # Dup
        # ----------------------------------------------------
        case jvm.Dup(words=w):
            if w == 1:
                [value], after = state.pop()

                yield (
                    pc + 1,
                    after.push(value).push(value),
                )
                return

            if w == 2:
                [first, second], after = state.pop(2)

                yield (
                    pc + 1,
                    after
                    .push(first)
                    .push(second)
                    .push(first)
                    .push(second),
                )
                return

            raise NotImplementedError(
                f"Dup with words={w} not supported"
            )
        # ----------------------------------------------------
        # Cast
        # ----------------------------------------------------

        case jvm.Cast():

            [value], after = state.pop()

            yield pc + 1, after.push(value)
            return

        # ----------------------------------------------------
        # Negate
        # ---------------------------------------------------

        case jvm.Negate(type=jvm.Int()):
            [value], after = state.pop()
            value = int_value(value)

            if value.values is not None:
                negated_vals = {-v for v in value.values}
                yield pc + 1, after.push(IntAbs.from_values(negated_vals))
            else:
                # Invierte los signos (+1 -> -1, -1 -> +1, 0 -> 0)
                negated_signs = {-s for s in value.signs.signs}
                yield pc + 1, after.push(IntAbs(None, SignSet(frozenset(negated_signs))))
            return

        case _:
            raise NotImplementedError(
                f"Unhandled static opcode at {pc}: {opr!r}"
            )

# ============================================================
# Initial state
# ============================================================

def initialstate(
    bc: jpamb.Bytecode,
    methodid: jvm.AbsMethodID,
    inputs: jpamb.Input | None,
) -> dict[PC, State]:

    method = bc.getmethod(methodid)

    locals = [IntAbs.bot()] * method.max_locals
    heap: dict[str, AbstractObject] = {}

    if inputs is None:
        for i, param in enumerate(methodid.extension.params):
            if isinstance(param, (jvm.Int, jvm.Boolean, jvm.Byte, jvm.Short, jvm.Char, jvm.Long, jvm.Float, jvm.Double)):
                locals[i] = IntAbs.top()

            elif isinstance(param, jvm.Array):
                token = f"arg-array:{i}"
                heap[token] = ArrayAbs.unknown(param.contains)
                locals[i] = RefAbs(frozenset({token}), True)

            elif (
                isinstance(param, jvm.ClassName)
                and str(param) == "java.lang.String"
            ):
                token = f"arg-string:{i}"
                heap[token] = StringAbs.top()
                locals[i] = RefAbs(frozenset({token}), True)

            else:
                locals[i] = RefAbs.unknown(f"arg:{i}")
    else:

        for i, value in enumerate(inputs.values):

            match value:

                case jpamb.case.Boolean(value=value):
                    locals[i] = IntAbs.concrete(
                        int(value)
                    )

                case jpamb.case.Int(value=value):
                    locals[i] = IntAbs.concrete(value)

                case jpamb.case.Array(
                    contains=element_type,
                    values=values,
                ):
                    token = f"arg-array:{i}"

                    if isinstance(element_type, jvm.Char):
                        concrete_values = [
                            ord(x)
                            for x in values
                        ]
                    else:
                        concrete_values = list(values)

                    heap[token] = ArrayAbs.concrete(
                        element_type,
                        concrete_values,
                    )

                    locals[i] = RefAbs(
                        frozenset({token}),
                        False,
                    )

                case jpamb.case.String(value=value):
                    token = f"arg-string:{i}"
                    heap[token] = StringAbs.concrete(value)

                    locals[i] = RefAbs(
                        frozenset({token}),
                        False,
                    )

                case _:
                    raise NotImplementedError(
                        f"Unsupported input value {value!r}"
                    )

    return {
        PC(methodid, 0): State(
            tuple(locals),
            (),
            tuple(sorted(heap.items())),
            (),
        )
    }


# ============================================================
# Worklist interpreter
# ============================================================

@dataclass
class AbstractInterpreter:
    bc: jpamb.Bytecode
    worklist: deque[PC]
    states: dict[PC, State]

    @staticmethod
    def initial(
        bc: jpamb.Bytecode,
        methodid: jvm.AbsMethodID,
        inputs,
    ):
        states = initialstate(
            bc,
            methodid,
            inputs,
        )

        return AbstractInterpreter(
            bc,
            deque(states.keys()),
            states,
        )

    def step(self) -> tuple[PC, set[str]]:
        pc = self.worklist.pop()

        print(
            f"Stepping {pc}:\n > {self.bc[pc]}",
            file=sys.stderr,
        )

        finals: set[str] = set()

        for result in manystep(
            self.bc,
            pc,
            self.states[pc],
        ):

            if isinstance(result, str):
                finals.add(result)
                continue

            pc_next, state = result

            before = self.states.get(pc_next)

            if before is None:
                after = state
            else:
                after = before | state

            if before is None or after != before:
                self.states[pc_next] = after
                if pc_next not in self.worklist:
                    self.worklist.append(pc_next)
            elif isinstance(self.bc[pc], jvm.Goto):
                finals.add("*")

        return pc, finals


# ============================================================
# JPAMB entry points
# ============================================================

def interpret():
    methodid, input_data, steps = jpamb.getcase(
        "October Abstract Interpreter",
        "1.0",
        "One hundred years of Analyzers",
        ["static", "python"],
        for_science=True,
    )

    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(
        suite,
        eff,
        {},
    )

    ai = AbstractInterpreter.initial(
        bc,
        methodid,
        input_data,
    )

    x = jpamb.emit_init(ai.states)

    while steps > 0 and ai.worklist:

        pc, final = ai.step()

        for value in final:
            jpamb.emit_step(
                x,
                pc,
                value,
                depth=1,
            )
            steps -= 1

        x = jpamb.emit_step(
            x,
            pc,
            ai.states,
            depth=1,
        )

        steps -= 1


def analyse():
    methodid = jpamb.getmethodid(
        "October Static Analizer",
                "1.0",
                "One hundred years of Analyzers",
                ["static", "python"],
        for_science=True,
    )

    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(suite, eff, {})
    ai = AbstractInterpreter.initial(bc, methodid, None)

    steps = 10000  # Aumenta ligeramente el límite para permitir convergencia de punto fijo
    final: set[str] = set()

    while steps > 0 and ai.worklist:
        _, finals = ai.step()
        final |= finals
        steps -= 1

    # Si la worklist se vacía, alcanzamos el punto fijo completo del análisis.
    reached_fixpoint = len(ai.worklist) == 0

    for query in jpamb.QUERIES:
        if query in final:
            # Si solo hubo este resultado y terminamos en punto fijo, es 'yes'
            if len(final) == 1 and reached_fixpoint:
                print(f"{query};yes")
            else:
                print(f"{query};maybe")
        else:
            # Si el análisis alcanzó punto fijo y el error jamás apareció en ningún estado
            if reached_fixpoint:
                print(f"{query};no")
            else:
                print(f"{query};maybe")
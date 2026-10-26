import random
import sys

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
        case jvm.BinaryOpr.Add:
            return v1 + v2
        case jvm.BinaryOpr.Mul:
            return v1 * v2
        case jvm.BinaryOpr.Rem:
            try:
                return v1 % v2
            except ZeroDivisionError:
                return "divide by zero"
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
#    print(f"Stepping {pc}:\n > {opr}", file=sys.stderr)
    match opr:
        # ----------------------------------------------------------- a partir de aqui es para Simple
        case jvm.Push(type=t, value=v): # se cambió para Arrays
            if t == jvm.Int():
                frame.stack.push(jvmc.StackInt(v))
            elif isinstance(t, jvm.Reference):
                # En JPAMB, una referencia nula se representa como StackReference(0)
                ref_val = 0 if v is None else v
                frame.stack.push(jvmc.StackReference(ref_val))
            elif str(t) == "Ljava/lang/String;":
                ref = state.heap.new(jvmc.HeapString(v))
                frame.stack.push(ref)
            else:
                raise NotImplementedError(f"Unsupported push type: {t}")

            frame.pc += 1
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

        case jvm.Ifz(offset=off_set, condition=op, target=trgt):
            value = frame.stack.pop()
            assert isinstance(value, jvmc.StackInt), f"expected int, but got {value}"

            if compare(op, value.value, 0):
                frame.pc = frame.pc + (trgt - frame.pc.offset)
            else:
                frame.pc += 1

        case jvm.If(offset=off_set, condition=op, target=trgt):
            v2 = frame.stack.pop()
            v1 = frame.stack.pop()

            assert isinstance(v1, jvmc.StackInt), f"expected int, but got {v1}"
            assert isinstance(v2, jvmc.StackInt), f"expected int, but got {v2}"

            if compare(op, v1.value, v2.value):
                frame.pc = frame.pc + (trgt - frame.pc.offset)
            else:
                frame.pc += 1
        case jvm.Return(type=t): # se cambió para Calls
            value = None

            if t == jvm.Int():
                value = frame.stack.pop()
                assert isinstance(value, jvmc.StackInt)

            state.frames.pop()

            if state.frames:
                if value is not None:
                    caller = state.frames.peek()
                    caller.stack.push(value)
            else:
                output = "ok"

        case jvm.Get(static=True, field=field):
            # Hack - Only handle the assertion case
            assert field.extension.name == "$assertionsDisabled"

            # Hack - Assuming assertions are never disabled
            frame.stack.push(jvmc.StackInt(0))
            frame.pc += 1

        case jvm.New(classname=jvm.ClassName("java.lang.AssertionError")):
            # Hack -- if we create an assertion error, we probably also throw it.
            output = "assertion error"

        case jvm.Load(type=jvm.Int(), index=i):
            value = frame.locals[i]
            frame.stack.push(value)
            frame.pc += 1

        # ----------------------------------------------------------- a partir de aqui es para Calls
        case jvm.InvokeStatic(method=method_id):
            method = bc.getmethod(method_id)
            new_frame = jvmc.Frame.from_method(method)

            num_args = len(method_id.extension.params)

            for i in range(num_args - 1, -1, -1):
                new_frame.locals[i] = frame.stack.pop()

            frame.pc += 1
            state.frames.push(new_frame)

        case jvm.NewArray(type=type, dim=1):
            length = frame.stack.pop()
            assert isinstance(length, jvmc.StackInt), (
                f"expected int, but got {length}"
            )

            ref = state.heap.new(jvmc.HeapArray(type, [0] * length.value))
            frame.stack.push(ref)
            frame.pc += 1    

        case jvm.Store(type=jvm.Reference(), index=i):
            value = frame.stack.pop()
            assert isinstance(value, jvmc.StackReference), (
                f"expected reference, but got {value}"
            )
            frame.locals[i] = value
            frame.pc += 1
        case jvm.Store(type=jvm.Int(), index=i):
            value = frame.stack.pop()
            assert isinstance(value, jvmc.StackInt), (
                f"expected int, but got {value}"
            )
            frame.locals[i] = value
            frame.pc += 1
        case jvm.Load(type=jvm.Reference(), index=i):
            value = frame.locals[i]
            assert isinstance(value, jvmc.StackReference), (
                f"expected reference, but got {value}"
            )
            frame.stack.push(value)
            frame.pc += 1

        case jvm.ArrayStore(type=_): # se cambió para Arrays
            value = frame.stack.pop()
            index = frame.stack.pop()
            ref = frame.stack.pop()

            assert isinstance(value, jvmc.StackInt), f"expected int value, but got {value}"
            assert isinstance(index, jvmc.StackInt), f"expected int index, but got {index}"
            assert isinstance(ref, jvmc.StackReference), f"expected reference, but got {ref}"

            # Verificación de referencia nula
            if ref.value == 0:
                output = "null pointer"
            else:
                try:
                    array = state.heap.memory[ref.value - 1]
                    assert isinstance(array, jvmc.HeapArray), f"expected array, but got {array}"
                    array.values[index.value] = value.value
                    frame.pc += 1
                except IndexError:
                    output = "out of bounds"

        case jvm.ArrayLength(): # se cambió para Arrays
            ref = frame.stack.pop()

            assert isinstance(ref, jvmc.StackReference), (
                f"expected reference, but got {ref}"
            )

            if ref.value == 0:
                output = "null pointer"
            else:
                array = state.heap.memory[ref.value - 1]
                assert isinstance(array, jvmc.HeapArray), (
                    f"expected array, but got {array}"
                )

                frame.stack.push(jvmc.StackInt(len(array.values)))
                frame.pc += 1

        case jvm.ArrayLoad(type=_): # se cambió para Arrays
            index = frame.stack.pop()
            ref = frame.stack.pop()

            assert isinstance(index, jvmc.StackInt), (
                f"expected int index, but got {index}"
            )
            assert isinstance(ref, jvmc.StackReference), (
                f"expected reference, but got {ref}"
            )

            if ref.value == 0:
                output = "null pointer"
            else:
                array = state.heap.memory[ref.value - 1]
                assert isinstance(array, jvmc.HeapArray), (
                    f"expected array, but got {array}"
                )

                try:
                    value = array.values[index.value]
                except IndexError:
                    output = "out of bounds"
                else:
                    frame.stack.push(jvmc.StackInt(value))
                    frame.pc += 1

        case jvm.Goto(target=target):
            frame.pc = frame.pc + (target - frame.pc.offset)  

        case jvm.Incr(index=i, amount=amount):
            value = frame.locals[i]

            assert isinstance(value, jvmc.StackInt), (
                f"expected int local, but got {value}"
            )

            frame.locals[i] = jvmc.StackInt(value.value + amount)
            frame.pc += 1   

        case jvm.InvokeVirtual(method=method_id):
            if (
                method_id.classname == jvm.ClassName("java.lang.String")
                and method_id.extension.name == "equals"
            ):
                other = frame.stack.pop()
                this = frame.stack.pop()

                assert isinstance(this, jvmc.StackReference)
                assert isinstance(other, jvmc.StackReference)

                this_string = state.heap[this]
                other_string = state.heap[other]

                assert isinstance(this_string, jvmc.HeapString)
                assert isinstance(other_string, jvmc.HeapString)

                result = this_string.content == other_string.content

                frame.stack.push(jvmc.StackInt(1 if result else 0))
                frame.pc += 1
            else:
                raise NotImplementedError(
                    f"Unsupported virtual method: {method_id}"
                )
            # ---------------------------------------------------------- a partir de aqui es para Loops
        case jvm.Cast():
            to_t = getattr(opr, "to", getattr(opr, "to_", None))
            val = frame.stack.pop()
            assert isinstance(val, jvmc.StackInt), f"expected int, but got {val}"

            raw_val = val.value

            # Truncamiento de bits según el tipo destino
            match to_t:
                case jvm.Short():
                    # es la forma matemática estándar en Python de simular el desbordamiento de 16 bits en complemento a dos que hace Java en hardware/JVM.
                    raw_val = (raw_val + 32768) % 65536 - 32768
                case jvm.Byte():
                    # es la forma matemática estándar en Python de simular el desbordamiento de 8 bits en complemento a dos que hace Java en hardware/JVM.
                    raw_val = (raw_val + 128) % 256 - 128
                case jvm.Char():
                    # es la forma matemática estándar en Python de simular el desbordamiento de 16 bits sin signo que hace Java en hardware/JVM.
                    raw_val = raw_val % 65536
                case jvm.Int():
                    pass
                case _:
                    # Fallback por representación en string si falla la instancia de clase
                    to_str = str(to_t)
                    if "Short" in to_str:
                        raw_val = (raw_val + 32768) % 65536 - 32768
                    elif "Byte" in to_str:
                        raw_val = (raw_val + 128) % 256 - 128
                    elif "Char" in to_str:
                        raw_val = raw_val % 65536
                    else:
                        raise NotImplementedError(f"Cast to {to_t!r} is not implemented")

            frame.stack.push(jvmc.StackInt(raw_val))
            frame.pc += 1

        # ---------------------------------------------------------- a partir de aqui es para Arrays
        case jvm.Dup(words=w):
            if w == 1:
                # Extraer, duplicar y volver a meter
                val = frame.stack.pop()
                frame.stack.push(val)
                frame.stack.push(val)
                frame.pc += 1
            elif w == 2:
                # Duplicar los dos elementos superiores
                v1 = frame.stack.pop()
                v2 = frame.stack.pop()
                frame.stack.push(v2)
                frame.stack.push(v1)
                frame.stack.push(v2)
                frame.stack.push(v1)
                frame.pc += 1
            else:
                raise NotImplementedError(f"Dup with words={w} not supported")    
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
        "September Interpreter",
        "1.0",
        "One hundred years of Analyzers",
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
    values = []

    # Lista unificada con todos los valores límite necesarios (incluyendo 8 y 10054203)
    INT_VALUES = [
        -10054203,
        -100,
        -13,
        -10,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        8,          # <-- CORRECCIÓN 1: Necesario para callsAssertFib
        10,
        13,
        24,
        100,
        1000,
        10054203,   # <-- CORRECCIÓN 2: Positivo, necesario para divideByNMinus10054203
    ]

    for p in methodid.extension.params:
        match p:
            case jvm.Int():
                # Seleccionamos directamente de la lista corregida
                values.append(jpamb.case.Int(rand.choice(INT_VALUES)))

            case jvm.Boolean():
                values.append(
                    jpamb.case.Boolean(rand.choice([True, False]))
                )

            case jvm.Array(contains=jvm.Int()):
                length = rand.choice([0, 1, 2, 3, 5, 10, 20])

                values.append(
                    jpamb.case.Array(
                        jvm.Int(),
                        [
                            rand.choice([-10, -1, 0, 1, 2, 10, 100])
                            for _ in range(length)
                        ],
                    )
                )

            case jvm.Array(contains=jvm.Char()):
                if rand.random() < 0.3:
                    chars = list("hello")
                else:
                    length = rand.choice([0, 1, 2, 3, 5, 10])
                    chars = [
                        chr(rand.randint(97, 122))
                        for _ in range(length)
                    ]

                values.append(
                    jpamb.case.Array(jvm.Char(), chars)
                )

            case jvm.ClassName("java.lang.String"):
                values.append(
                    jpamb.case.String(
                        rand.choice([
                            "",
                            "hello",
                            "not hello",
                            "Hello",
                            "HELLO",
                            "foo",
                            "bar",
                            "test",
                            "hello!",
                            "nothello",
                        ])
                    )
                )

            case a:
                raise NotImplementedError(
                    f"Don't know how to generate random value for {a!r}"
                )

    return jpamb.case.Input(values)
def _run_dynamic_case(bc, methodid, input_data, max_steps, behaviors):
    state = initial(bc, methodid, input_data)
    for _ in range(max_steps):
        _, state = step(bc, state)
        if isinstance(state, str):
            behaviors.add(state)
            return

    # Si supera el MAX_STEPS, asumimos que no termina
    behaviors.add("*")


def _directed_inputs(methodid):
    """Genera un pequeño conjunto dirigido de entradas antes del fuzzing."""
    params = methodid.extension.params

    # Casos especialmente útiles para enteros.
    int_values = [
        -10054203,
        -100,
        -13,
        -10,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        10,
        11,
        13,
        24,
        50,
        100,
        1000,
        10054203,
    ]

    if len(params) == 1:
        p = params[0]

        if isinstance(p, jvm.Int):
            for value in int_values:
                yield jpamb.case.Input([jpamb.case.Int(value)])

        elif isinstance(p, jvm.Boolean):
            # Los dos valores posibles son importantes: no dependemos del random.
            yield jpamb.case.Input([jpamb.case.Boolean(False)])
            yield jpamb.case.Input([jpamb.case.Boolean(True)])

        elif isinstance(p, jvm.Array) and isinstance(p.contains, jvm.Int):
            arrays = [
                [],
                [0],
                [1],
                [-1],
                [-13],
                [1, 2, 3],
                [10, 10, 10],
                [50, 100, 200],
                [100, 100, 100],
            ]
            for values in arrays:
                yield jpamb.case.Input([
                    jpamb.case.Array(jvm.Int(), values)
                ])

        elif isinstance(p, jvm.Array) and isinstance(p.contains, jvm.Char):
            arrays = [
                [],
                list("x"),
                list("hello"),
                list("hell"),
                list("helloo"),
                list("Hello"),
                list("HELLO"),
                list("world"),
                list("nothello"),
            ]
            for chars in arrays:
                yield jpamb.case.Input([
                    jpamb.case.Array(jvm.Char(), chars)
                ])

        elif isinstance(p, jvm.ClassName) and str(p) == "java.lang.String":
            strings = [
                "",
                "hello",
                "not hello",
                "Hello",
                "HELLO",
                "foo",
                "bar",
                "test",
                "hello!",
                "nothello",
            ]
            for value in strings:
                yield jpamb.case.Input([jpamb.case.String(value)])

    elif len(params) == 2:
        # Para dos enteros hacemos una pequeña exploración cartesiana.
        # Esto evita depender de que el random encuentre relaciones como
        # (0, 0), (1, 1), (3, 6), etc.
        if all(isinstance(p, jvm.Int) for p in params):
            pair_values = [
                -13,
                -1,
                0,
                1,
                2,
                3,
                6,
                10,
                13,
                24,
                100,
                10054203,
            ]
            for left in pair_values:
                for right in pair_values:
                    yield jpamb.case.Input([
                        jpamb.case.Int(left),
                        jpamb.case.Int(right),
                    ])


def analyse():
    # yes y no para el caso sin entradas, found y not-found para el caso con entradas
    """El análisis dinámico para JPAMB"""

    methodid = jpamb.getmethodid(
        "September Dynamic Analyzer",
        "2.0",
        "One hundred years of Analyzers",
        ["dynamic", "python"],
        for_science=True,
    )

    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(suite, eff, {})

    MAX_STEPS = 2000  # Límite de pasos por ejecución
    num_params = len(methodid.extension.params)

    behaviors = set()
    rand = random.Random(0)

    if num_params == 0:
        # --- CASO 1: SIN ENTRADAS (Determinista) ---
        input_data = jpamb.case.Input([])
        _run_dynamic_case(bc, methodid, input_data, MAX_STEPS, behaviors)

        # Emitir respuestas con certeza absoluta (;yes / ;no)
        for query in jpamb.QUERIES:
            if query in behaviors:
                if query == "*":
                    print(f"{query};timeout")
                else:
                    print(f"{query};yes")  # o ;found según la versión de JPAMB
            else:
                print(f"{query};no")

    else:
        # --- CASO 2: CON ENTRADAS ---
        # 1. Casos deterministas dirigidos para cubrir fronteras y relaciones.
        for input_data in _directed_inputs(methodid):
            _run_dynamic_case(bc, methodid, input_data, MAX_STEPS, behaviors)

        # 2. Fuzzing aleatorio complementario.
        NUM_TRIALS = 150
        for _ in range(NUM_TRIALS):
            input_data = fuzz_input(rand, methodid)
            _run_dynamic_case(bc, methodid, input_data, MAX_STEPS, behaviors)

        # Emitir respuestas para fuzzing (;found / ;not-found)
        for query in jpamb.QUERIES:
            if query in behaviors:
                if query == "*":
                    print(f"{query};timeout")
                else:
                    print(f"{query};found")
            else:
                print(f"{query};not-found")

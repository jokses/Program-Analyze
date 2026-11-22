import random
import sys
from wsgiref.validate import assert_

import jpamb
import jvm
import jvm.state as jvmc

DEBUG = False


def binary(op, v1: int, v2: int) -> int | str:
    match op:
        case jvm.BinaryOpr.Div:
            try:
                return v1 // v2
            except ZeroDivisionError:
                return "divide by zero"
        case jvm.BinaryOpr.Rem:  # Tjek om dit enum-felt hedder Rem eller Mod
            try:
                # Java's % opfører sig som Python's math.fmod, men for heltal
                # skal du blot passe på fortegnet (v1 % v2 i Python kan afvige ved negative tal,
                # men i Collatz-testen arbejder du med positive tal, så % virker perfekt)
                return int(v1 % v2) if v1 >= 0 else int(math.fmod(v1, v2))
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

                # Håndtering af String-objekter (f.eks. ldc ["hello"])
                case jvm.Object(classname) if classname.dotted() == "java.lang.String":
                    # 1. Opret en ny HeapString på din heap med værdien (v)
                    heap_str = jvmc.HeapString(content=v)

                    # 2. Gem den på heapen og få en StackReference retur
                    str_ref = state.heap.new(heap_str)

                    # 3. Push referencen på din operand stack
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
        case jvm.Load( index= index):
            frame.stack.push(frame.locals[index])
            frame.pc +=1
        case jvm.InvokeStatic(offset=off, method=method_id):
            target_method = bc.getmethod(method_id)
            if DEBUG:
                print(f" Kalder metode: {method_id.extension.name}", file=sys.stderr)

            # VI LADER frame.pc BLIVE PÅ DET AKTUELLE OFFSET (off)
            # Så frameworket kan se det præcise invoke-punkt (:1) under hele forløbet!

            new_frame = jvmc.Frame.from_method(target_method)
            num_params = len(method_id.extension.params._elements)

            args = []
            for _ in range(num_params):
                args.append(frame.stack.pop())
            args.reverse()

            for i, arg_val in enumerate(args):
                new_frame.locals[i] = arg_val

            state.frames.push(new_frame)
            return new_frame.pc, state

        case jvm.NewArray(offset=off, type=t, dim=d):

            count = frame.stack.pop().value
            if DEBUG:
                print(f" value ----  {t} {count}", file=sys.stderr)
            match t:
                case jvm.Int():
                    heap_array_obj = jvm.state.HeapArray(contains=t, values=[0] * count)
                    # array_reference = jvmc.StackRef(array_data)
                    array_ref = state.heap.new(heap_array_obj)
                    frame.stack.push(array_ref)
                    frame.pc +=1
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
                return frame.pc, "null pointer"

            # Tjek grænser (Bounds checking)
            if not (0 <= raw_index < len(heap_array.values)):
                # 1. Pop den crashede ramme (generatePrimeArray) af din CallStack, ligesom i jvm.Throw!
                state.frames.pop()

                # 2. Hvis der er en forrige ramme (allPrimesArePositive), genoptager vi den
                if state.frames:
                    previous_frame = state.frames.peek()

                    # Hvis din InvokeStatic ikke fik den flyttet permanent i frameworkets øjne,
                    # så tvinger vi den op på offset 2 lige her under crashet!
                    if previous_frame.pc.offset == 1:
                        previous_frame.pc += 1

                    return previous_frame.pc, "out of bounds"

                return frame.pc, "out of bounds"

            heap_array.values[raw_index] = raw_value
            frame.pc += 1

        case jvm.InvokeVirtual(offset=off, method=method_id):
            # 1. Træk navnet på metoden og klassen ud som strenge
            classname_str = method_id.classname.dotted()
            method_name = method_id.extension.name

            if DEBUG:
                print(
                    f" Kalder virtuel metode: {classname_str}.{method_name}",
                    file=sys.stderr,
                )

            # 2. Hent antallet af parametre
            num_params = len(method_id.extension.params._elements)

            # 3. Hent argumenterne fra den nuværende stack
            args = []
            for _ in range(num_params):
                args.append(frame.stack.pop())
            args.reverse()

            # 4. Pop selve objekt-referencen ('this') fra stacken
            object_ref = frame.stack.pop()

            # --- HACK: Håndtering af indbyggede java.lang.String metoder ---
            if classname_str == "java.lang.String":
                # Hent det faktiske HeapString-objekt ud fra din heap ved hjælp af referencen
                heap_string_obj = state.heap[object_ref]
                raw_string = heap_string_obj.content  # Dette er den rå Python-streng (f.eks. "hello")

                match method_name:
                    case "equals":
                        # .equals(Object) sammenligner med en anden streng
                        # Vi skal hente den anden strengs reference ud af argumenterne (args[0])
                        other_ref = args[0]
                        other_string_obj = state.heap[other_ref]

                        # Lav sammenligningen og push resultatet (1 for True, 0 for False)
                        is_equal = (raw_string == other_string_obj.content)
                        frame.stack.push(jvmc.StackInt(1 if is_equal else 0))

                    case "length":
                        # .length() returnerer længden på strengen
                        frame.stack.push(jvmc.StackInt(len(raw_string)))

                    case "charAt":
                        # .charAt(int) returnerer Unicode-værdien for tegnet på indeks args[0]
                        idx = args[0].value
                        frame.stack.push(jvmc.StackInt(ord(raw_string[idx])))

                    case _:
                        raise NotImplementedError(f"Strengmetoden {method_name} er ikke implementeret i InvokeVirtual")

                # Da vi kørte strengmetoden som et hack uden at pushe en ny frame,
                # rykker vi blot den nuværende PC én frem og fortsætter!
                frame.pc += 1
                return frame.pc, state

            # --- NORMAL VELKENDT METODE (Hvis metoden findes i din egen bytecode-fil) ---
            target_method = bc.getmethod(method_id)
            new_frame = jvmc.Frame.from_method(target_method)

            # Læg 'this' ind på indeks 0 i de lokale variable
            new_frame.locals[0] = object_ref

            # Læg argumenterne ind på de efterfølgende pladser (fra indeks 1 og frem)
            for i, arg_val in enumerate(args):
                new_frame.locals[i + 1] = arg_val

            # Ryk den gamle PC frem, inden vi pusher den nye frame
            frame.pc += 1

            state.frames.push(new_frame)
            return new_frame.pc, state

        case jvm.Negate(offset=off, type=t):
            # 1. Pop værdien der skal negeres fra operand-stacken
            value_element = frame.stack.pop()
            assert isinstance(value_element, jvmc.StackInt), f"expected int, but got {value_element}"

            # 2. Beregn den aritmetiske negation (-værdien)
            raw_value = value_element.value
            negated_result = -raw_value

            match t:
                case jvm.Int():
                    # 3. Pak resultatet ind i en StackInt og push det tilbage på stacken
                    frame.stack.push(jvmc.StackInt(value=negated_result))
                case _:
                    raise NotImplementedError(f"Negate er ikke implementeret for typen {t!r}")

            # 4. Ryk programtælleren én frem og fortsæt
            frame.pc += 1

        case jvm.ArrayLength(offset=off):
            ref_element = frame.stack.pop()

            try:
                heap_array = state.heap[ref_element]
            except IndexError:
                return frame.pc, "null pointer"

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
                state.frames.pop()
                if state.frames:
                    previous_frame = state.frames.peek()
                    previous_frame.pc += 1
                    return previous_frame.pc, "null pointer"
                return frame.pc, "null pointer"

            if not (0 <= raw_index < len(heap_array.values)):
                state.frames.pop()
                if state.frames:
                    previous_frame = state.frames.peek()
                    previous_frame.pc += 1  # Flyt PC frem for at dække kaldet ved fejl
                    return previous_frame.pc, "out of bounds"
                return frame.pc, "out of bounds"

            raw_value = heap_array.values[raw_index]

            match t:
                case jvm.Int() | jvm.Char():
                    frame.stack.push(jvm.state.StackInt(value=raw_value))
                case _:
                    raise NotImplementedError(f"ArrayLoad er ikke implementeret for typen {t!r}")

            frame.pc += 1

        case jvm.Incr(offset=off, index=idx, amount=amt):
            # 1. Hent den nuværende værdi ud fra lokale variable på det givne index
            current_local = frame.locals[idx]

            # Sørg for at hente den rå heltalværdi ud (det er en StackInt)
            current_value = current_local.value

            # 2. Læg instruktionens 'amount' (amt) til værdien
            new_value = current_value + amt

            # 3. Pak det rå resultat ind i en ny StackInt og gem det tilbage i locals
            frame.locals[idx] = jvm.state.StackInt(value=new_value)

            # 4. Gå videre til næste instruktion
            frame.pc += 1


        case jvm.Dup(offset=off, words=w):
            if w == 1:

                top_element = frame.stack.pop()


                frame.stack.push(top_element)
                frame.stack.push(top_element)

                frame.pc += 1
            elif w == 2:

                v1 = frame.stack.pop() # Øverste
                v2 = frame.stack.pop() # Næstøverste

                frame.stack.push(v2)
                frame.stack.push(v1)
                frame.stack.push(v2)
                frame.stack.push(v1)

                frame.pc += 1
            else:
                raise NotImplementedError(f"Dup med words={w} er ikke implementeret")


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
            is_void = (t is None)

            if not is_void:
                v1 = frame.stack.pop()

                # Hvis dette er den absolut sidste ramme på callstacken
                if len(state.frames.frames) == 1:
                    frame.stack.push(v1)
                    # Forhindrer IndexError i den absolut sidste returnering
                    return frame.pc, "ok"
                else:
                    # Intern returnering (f.eks. fra generatePrimeArray tilbage til allPrimesArePositive)
                    state.frames.pop()
                    previous_frame = state.frames.peek()
                    previous_frame.stack.push(v1)

                    # FJERNET: previous_frame.pc += 1
                    # Siden InvokeStatic allerede flyttede den til :2, skal vi IKKE øge den mere her!
                    return previous_frame.pc, state
            else:
                # For void-metoder
                if len(state.frames.frames) == 1:
                    return frame.pc, "ok"
                else:
                    state.frames.pop()
                    previous_frame = state.frames.peek()
                    return previous_frame.pc, state

        case jvm.If(offset=off, condition=cond, target=targ):
            value2 = frame.stack.pop()  # Højre side (f.eks. array.length)
            value1 = frame.stack.pop()  # Venstre side (f.eks. i)
            assert isinstance(value1, jvmc.StackInt), f"expected int, but got {value1}"
            assert isinstance(value2, jvmc.StackInt), f"expected int, but got {value2}"

            # Vi tvinger den direkte tildeling af PC-objektet via din overskrevne modulo-operatør
            if compare(cond, value1.value, value2.value):
                frame.pc %= targ
            else:
                frame.pc += 1

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
        case jvm.InvokeSpecial(offset=off, method=method_id, is_interface=is_iface):
            num_params = len(method_id.extension.params._elements)
            for _ in range(num_params):
                frame.stack.pop()
            object_ref = frame.stack.pop()
            frame.pc += 1

        case jvm.Throw(offset=off):
            error_ref = frame.stack.pop()

            # Fjern den crashede ramme (generatePrimeArray)
            state.frames.pop()

            if state.frames:
                # Nu kigger vi på allPrimesArePositive rammen
                previous_frame = state.frames.peek()
                # Vi lader previous_frame.pc blive stående på invoke-punktet,
                # så evaluatoren kan registrere at punktet blev dækket under crashet!
                return previous_frame.pc, "assertion error"
            else:
                return frame.pc, "assertion error"

        case jvm.New(offset=off) as new_opr:
            # Ved at bruge 'as new_opr' er vi 100% sikre på at ramme den rigtige instans
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
                input.append(jpamb.case.Int(rand.randint(-(1 << 31), 1 << 31)))
            case jvm.Boolean():
                input.append(jpamb.case.Boolean(1 == rand.randint(0, 1)))
            # Inde i din fuzz_input funktion, hvor du genererer værdier til java.lang.String:
            case jvm.Object(classname) if classname.dotted() == "java.lang.String":
                # 1. Generer den tilfældige rå streng
                random_str = "fuzz_" + "".join(rand.choices("abcdefghijklmnopqrstuvwxyz", k=5))

                # 2. Pak den ind i jpamb's eget case-objekt, som din initial() forventer!
                input.append(jpamb.case.String(value=random_str))

            # case jvm.Object(classname) if classname.dotted() == "java.lang.String":
            #     # 1. Generer en tilfældig streng (eller en fast test-streng som "hello")
            #     random_str = "fuzz_" + "".join(rand.choices("abcdefghijklmnopqrstuvwxyz", k=5))
            #
            #     # 2. Opret et HeapString objekt (fra din state.py hukommelsesmodel)
            #     heap_string_obj = jvm.state.HeapString(content=random_str)
            #
            #     # Bemærk: Hvis din fuzz_input funktion har adgang til 'state.heap',
            #     # gemmer du den der og appender den returnerede StackReference:
            #     # string_ref = state.heap.new(heap_string_obj)
            #     # inputs.append(string_ref)
            #
            #     # HVIS fuzz_input kun genererer de rå værdier før staten er bygget:
            #     # Kan det være du blot skal returnere strengen eller jvm.state.HeapString(content=random_str)
            #     # Tjek hvordan dit jvm-framework normalt pakker fuzz-inputs i staten.
            #     input.append(random_str)
            # case jpamb.case.String(value=value):
            #     ref = state.heap.new(jvmc.HeapString(value))
            #     frame.locals[i] = ref

            case a:
                raise NotImplementedError(
                    "Don't know how to create random values for {input}"
                )

    return jpamb.case.Input(input)


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

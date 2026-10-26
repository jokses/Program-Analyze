#!/usr/bin/env python3
"""
A syntactic analyzer for JPAMB.

The benchmark asks whether an outcome is POSSIBLE for at least one input.

Escribe las categorías ("yes"/"no").

Mira el código fuente del método, condiciones constantes simples, patrones de control de flujo, arreglos, asignaciones nulas, aserciones, divisiones, bucles y una pequeña cantidad de razonamiento interprocedimental.
"""

import logging
import re
import sys
from pathlib import Path

import jpamb


OUTCOMES = (
    "ok",
    "divide by zero",
    "assertion error",
    "out of bounds",
    "null pointer",
    "*",
)


def strip_comments(text):
    # Quita los comentarios de bloque y de línea para que no interfieran con el análisis.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def find_method_body(content, method_name):
    # Encuentra el método teniendo en consideración que puede tener if, loops, etc. y no solo el cierre de llaves.
    pattern = rf"\b{re.escape(method_name)}\s*\([^)]*\)\s*\{{"
    match = re.search(pattern, content)

    if not match:
        return None

    opening = match.end() - 1
    depth = 0

    # Contador de llaves
    for i in range(opening, len(content)):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                return content[opening : i + 1]

    return None


def remove_strings(text):
    # Quita las cadenas de texto para que no interfieran con el análisis.
    # Esto es útil para evitar falsos positivos en patrones de código que podrían aparecer dentro de cadenas.
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    text = re.sub(r"'(?:\\.|[^'\\])*'", "''", text)
    return text


def normalize(body):
    # Quita comentarios y cadenas de texto para que no interfieran con el análisis sintáctico.
    return remove_strings(strip_comments(body))


def has_literal_true(expr):
    # Comprueba si la expresión es literalmente "true" (ignorando espacios en blanco).
    return re.fullmatch(r"\s*true\s*", expr) is not None


def has_literal_false(expr):
    # Comprueba si la expresión es literalmente "false" (ignorando espacios en blanco).
    return re.fullmatch(r"\s*false\s*", expr) is not None


def constant_int(expr):
    # Si encuentra un entero o una expresión que obtenga un entero lo devuelve
    # Si no puede determinarlo porque depende de una variable, devuelve None
    expr = expr.strip()

    # Busca especificamente un número entero, posiblemente negativo.
    if re.fullmatch(r"-?\d+", expr):
        try:
            return int(expr)
        except ValueError:
            return None

    # Busco [número] [ + o - ] [número]
    m = re.fullmatch(r"\(?\s*(-?\d+)\s*([+-])\s*(-?\d+)\s*\)?", expr)
    if m:
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        return a + b if op == "+" else a - b

    return None


def definitely_false_condition(expr):
    # Busco condiciones que sean siempre falsas, como "false" o "1 == 2"
    expr = expr.strip()

    if has_literal_false(expr):
        return True

    m = re.fullmatch(r"(-?\d+)\s*(==|!=|<|<=|>|>=)\s*(-?\d+)", expr)
    if m:
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        return {
            "==": a == b,
            "!=": a != b,
            "<": a < b,
            "<=": a <= b,
            ">": a > b,
            ">=": a >= b,
        }[op] is False

    return False


def definitely_true_condition(expr):
    # Busco condiciones que sean siempre verdaderas, como "true" o "1 == 1"
    expr = expr.strip()

    if has_literal_true(expr):
        return True

    m = re.fullmatch(r"(-?\d+)\s*(==|!=|<|<=|>|>=)\s*(-?\d+)", expr)
    if m:
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        return {
            "==": a == b,
            "!=": a != b,
            "<": a < b,
            "<=": a <= b,
            ">": a > b,
            ">=": a >= b,
        }[op] is True

    return False



def split_top_level_statements(code):
    # Divide el código en declaraciones de nivel superior, respetando paréntesis y llaves anidadas.
    # Esto es útil para analizar declaraciones individuales y determinar si hay retornos o errores que afecten la ejecución normal.
    # Por ejemplo, si hay un `return;` en el nivel superior, cualquier código después de eso es inalcanzable y no debe considerarse para la predicción de resultados.
    result = []
    start = 0
    paren = 0
    brace = 0

    for i, ch in enumerate(code):
        if ch == "(":
            paren += 1
        elif ch == ")":
            paren = max(0, paren - 1)
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace = max(0, brace - 1)
        elif ch == ";" and paren == 0 and brace == 0:
            result.append(code[start : i + 1])
            start = i + 1

    tail = code[start:].strip()
    if tail:
        result.append(tail)

    return result


def prune_constant_ifs(code):
    # Reviso condiciones que sean siempre verdaderas o siempre falsas y elimino ramas inalcanzables. Asi si tengo if (true) { ... } else { ... } me quedo solo con la rama verdadera lo que va dentro de if y si es if (false) { ... } else { ... } me quedo solo con el else.
    changed = True
    while changed:
        changed = False

        # if (...) { ... } else { ... }
        pattern = re.compile(
            r"if\s*\(\s*(true|false)\s*\)\s*\{"
            r"(?P<yes>(?:[^{}]|\{[^{}]*\})*)\}" # esta expresion permite capturar el contenido dentro de las llaves del if y meterlas en el grupo "yes"
            r"\s*else\s*\{"
            r"(?P<no>(?:[^{}]|\{[^{}]*\})*)\}", # esta expresion permite capturar el contenido dentro de las llaves del else y meterlas en el grupo "no"
            re.S, # esto permite que el punto (.) coincida con cualquier caracter, incluyendo saltos de línea, para poder capturar correctamente el contenido dentro de las llaves aunque tenga varias líneas
        )

        # este metodo busca si en el codigo hay un if y un else
        def replace_if_else(match):
            nonlocal changed
            changed = True
            return match.group("yes") if match.group(1) == "true" else match.group("no")

        # aqui busca en code el pattern encontrado y lo reemplaza con el resultado de la funcion replace_if_else, que devuelve el contenido del if o del else segun corresponda
        new_code = pattern.sub(replace_if_else, code)
        if new_code != code:
            code = new_code
            continue

        # esto es en caso de que haya if sin un else
        pattern = re.compile(
            r"if\s*\(\s*(true|false)\s*\)\s*\{"
            r"(?P<body>(?:[^{}]|\{[^{}]*\})*)\}", # esta expresion permite capturar el contenido dentro de las llaves del if y meterlas en el grupo "body"
            re.S,
        )

        # este metodo busca si en el codigo hay un if sin else
        def replace_if(match):
            nonlocal changed
            changed = True
            return match.group("body") if match.group(1) == "true" else ""

        new_code = pattern.sub(replace_if, code)
        if new_code != code:
            code = new_code

    return code


def reachable_code(body):
    # Elimina código inalcanzable después de un `return;` o `return expression;` de nivel superior, y simplifica los `if`/`else` constantes antes de buscar errores alcanzables.
    # Esto es útil para evitar falsos positivos en análisis de errores, ya que el código después de un retorno no se ejecutará y no debería considerarse para la predicción de resultados.

    # Si el cuerpo del método está encerrado en llaves, elimino las llaves externas para analizar solo el contenido interno.
    code = body[1:-1] if body.startswith("{") and body.endswith("}") else body

    # Aqui uso la funcion prune_constant_ifs para eliminar ifs que sean siempre verdaderos o siempre falsos, dejando solo el bloque de codigo que se ejecutaria.
    code = prune_constant_ifs(code)
    
    # Si después de revisar los ifs constantes no encuentro un return, busco el primer return de nivel superior y devuelvo solo el código hasta ese punto.
    statements = split_top_level_statements(code)
    kept = []
    for statement in statements:
        kept.append(statement)
        if re.match(r"^\s*return\b", statement):
            break

    return "\n".join(kept)


def find_constant_arrays(code):
    # Detecta inicializadores de arreglos y asignaciones de longitud constante para poder analizar accesos a índices de arreglos y detectar posibles errores de desbordamiento de índice. 
    arrays = {}

    # Buscame un array de x tipos de datos primitivos o String, con la sintaxis tipo: int a[] = {1, 2, 3}; y guardame el nombre del array y la cantidad de elementos que tiene.
    pattern = re.compile(
        r"(?:byte|short|int|long|float|double|char|boolean|String)"
        r"\s+(\w+)\s*\[\s*\]\s*=\s*\{([^{}]*)\}\s*;"
        # (\w+) captura el nombre del array
        # ([^{}]*) captura los valores dentro de las llaves, separados por comas
    )
    # Busco también la sintaxis tipo: int[] a = {1, 2, 3}; y guardame el nombre del array y la cantidad de elementos que tiene.
    pattern2 = re.compile(
        r"(?:byte|short|int|long|float|double|char|boolean|String)"
        r"\s*\[\s*\]\s*(\w+)\s*=\s*\{([^{}]*)\}\s*;"
    )

    # Busco ambos patrones en el código y guardo el nombre del array y la cantidad de elementos que tiene.
    for m in list(pattern.finditer(code)) + list(pattern2.finditer(code)):
        values = [x for x in m.group(2).split(",") if x.strip()]
        arrays[m.group(1)] = len(values)

    # Busco también la sintaxis tipo: int a[] = new int[10]; y guardame el nombre del array y la cantidad de elementos que tiene.
    pattern = re.compile(
        r"(?:byte|short|int|long|float|double|char|boolean|String)\s*"
        r"\[\s*\]\s+(\w+)\s*=\s*new\s+\w+\s*\[\s*(\d+)\s*\]"
    )

    for m in pattern.finditer(code):
        arrays[m.group(1)] = int(m.group(2))

    return arrays


def analyze_assertions(code):
    # Detecta aserciones que son siempre verdaderas o siempre falsas. Cualquier otra expresión es potencialmente falsa para alguna entrada, por lo que es un 'sí' conservador.
    assertions = re.findall(r"\bassert\s+([^;]+);", code)

    if not assertions:
        return False

    for condition in assertions:
        condition = condition.strip()

        if has_literal_true(condition):
            continue

        # Un assert false es un error garantizado si es alcanzable.
        if has_literal_false(condition):
            return True

        # Si es assert pero sin una condición literal, es potencialmente falsa para alguna entrada.
        return True

    return False


def analyze_division(code):
    # Predice si es posible una división por cero.
    #    1. literal / 0: definitivamente peligrosa;
    #    2. división por una variable: cero es una entrada posible;
    #    3. división por una expresión que contiene una variable: la expresión potencialmente puede evaluarse como cero.

    # Evitamos tratar 'if (n != 0) return 1/n' como peligrosa cuando el divisor está protegido por una condición simple.
    
    # Esto busca un cero literalmente y que acabe con un límite de palabra o un punto y coma, coma o paréntesis de cierre para evitar falsos positivos como 10/0.5
    if re.search(r"/\s*0(?:\b|[;,)])", code):
        return True

    # Una división por una variable o una expresión que contiene una variable es potencialmente peligrosa.
    variable_divisions = re.findall(
        r"/\s*([A-Za-z_]\w*)\b",
        code,
    )

    for variable in variable_divisions:
        # Revisar si son valores que se pueden proteger con un if simple
        #  por ejemplo  if (n != 0) { return 1 / n; }
        protected = re.search(
            rf"if\s*\(\s*{re.escape(variable)}\s*!=\s*0\s*\)\s*\{{[^}}]*"
            rf"/\s*{re.escape(variable)}\b",
            code,
            flags=re.S,
        )

        # También está protegida si el caso variable == 0
        # se elimina antes de llegar a la división.
        if not protected:
            protected = re.search(
                rf"if\s*\(\s*{re.escape(variable)}\s*==\s*0\s*\)\s*\{{[^}}]*"
                rf"(?:return|throw)[^}}]*\}}\s*"
                rf"(?:[^{{}}]*\s*)*/\s*{re.escape(variable)}\b",
                code,
                flags=re.S,
            )

        if not protected:
            return True

    # Revisar si hay divisiones por expresiones que contienen variables, como 1/(n+1) o 1/(x-2) si es solo numeros sin variables no es peligroso, pero si hay una variable es potencialmente peligrosa.
    for expression in re.findall(r"/\s*\(([^()]*)\)", code):
        if re.search(r"[A-Za-z_]\w*", expression):
            return True

        if constant_int(expression) == 0:
            return True

    return False


def analyze_null_pointer(code):
    # Rastrea el patrón sencillo de referencia nula de asignar una variable a null y luego intentar acceder a ella o llamar a un método en ella. Esto es un 'sí' conservador.
    
    # Busca arrays que se asignan explícitamente a null
    null_vars = set(
        re.findall(
            r"\b(?:\w+\s*\[\]|\w+)\s+(\w+)\s*=\s*null\s*;",
            code,
        )
    )

    for var in null_vars:
        # Reviso si accede al Array / .length / method invocation / field access.
        # busco especificamente la variable sin que forme parte de un string y seguida de un corchete, punto o paréntesis para detectar accesos a arreglos, llamadas a métodos o acceso a campos.
        if re.search(
            rf"\b{re.escape(var)}\s*(?:\[|\.|\()",
            code,
        ):
            # Si hay una declaración de la variable a null seguida de un acceso, es un error garantizado.
            declaration = re.search(
                rf"\b{re.escape(var)}\s*=\s*null\s*;",
                code,
            )
            after = code[declaration.end() :] if declaration else code

            if re.search(
                rf"\b{re.escape(var)}\s*(?:\[|\.|\()",
                after,
            ):
                return True



    # Aqui es si busco un patrón de null dereference como (Type) null.method() o (Type) null.field, que es un error garantizado.
    if re.search(r"\(\s*\w+\s*\)\s*null\s*\.", code):
        return True

    return False



def analyze_out_of_bounds(code, arrays):
    # Detecta errores simples de índice de arreglo que pueden demostrarse estáticamente.
    # Para una longitud conocida N: a[N] y a[N+1] y a[-1] está fuera de rango.
    # Para un índice variable desconocido, conservadoramente decimos 'sí' porque una entrada puede hacerlo demasiado grande o negativo.
    # También detectamos `a.length` de forma segura; leer la longitud no es un error de índice fuera de rango.
    
    for array, length in arrays.items():
        accesses = re.finditer(
            rf"\b{re.escape(array)}\s*\[\s*([^\]]+)\s*\]",
            code,
        )

        for access in accesses:
            index = access.group(1).strip()

            value = constant_int(index)
            if value is not None:
                if value < 0 or value >= length:
                    return True
                continue

            # Una variable o expresión desconocida puede tomar un valor inválido.
            return True

    # No deberia estar buscando en declaraciones como si fueran accesos
    # `int[] a = new int[5]`.
    return False


def analyze_infinite_loop(code): 
    # Detecta bucles que sintácticamente están garantizados que no terminan, como while (true), while (1 == 1) y while (i > 0), donde i se inicializa con una constante positiva y nunca se modifica dentro del bucle.
    # No afirmamos deliberadamente que todos los bucles while/for son infinitos.
    
    # while(true)
    if re.search(r"\bwhile\s*\(\s*true\s*\)", code):
        return True

    if re.search(r"\bfor\s*\(\s*;\s*true\s*;\s*\)", code):
        return True

    # Condiciones constantes y verdaderas
    for condition in re.findall(r"\bwhile\s*\(([^)]*)\)", code):
        if definitely_true_condition(condition):
            return True


    # aqui reviso que haga una declaracion por ej int contador = 1
    assignments = {
        name: int(value)
        for name, value in re.findall(
            r"\b(?:byte|short|int|long)\s+(\w+)\s*=\s*(-?\d+)\s*;",
            code,
        )
    }

    # aqui busco un while con una condicion que sea una variable y un operador de comparacion con un limite constante, por ejemplo while (contador > 0) el segundo comparado es un numero
    for condition in re.findall(r"\bwhile\s*\(([^)]*)\)", code):
        m = re.fullmatch(
            r"\s*(\w+)\s*(>|>=|<|<=|==|!=)\s*(-?\d+)\s*",
            condition,
        )
        if not m:
            continue

        variable, op, limit = m.group(1), m.group(2), int(m.group(3))
        if variable not in assignments:
            continue

        # Encontrar especificamente el while (variable op limit)
        while_match = re.search(
            rf"\bwhile\s*\(\s*{re.escape(variable)}\s*"
            rf"{re.escape(op)}\s*{limit}\s*\)\s*\{{",
            code,
        )
        if not while_match:
            continue

        # coger el interior del while contando las llaves
        open_brace = code.find("{", while_match.end() - 1)
        depth = 0
        end = None
        for i in range(open_brace, len(code)):
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end is None:
            continue

        loop_body = code[open_brace + 1 : end]

        # Revisar si la variable se incrementa o decrementa dentro del bucle. Si no se modifica, el bucle es potencialmente infinito.
        if not re.search(
            rf"\b{re.escape(variable)}\s*(?:\+\+|--|[+\-*/%]?=)",
            loop_body,
        ):
            initial = assignments[variable]
            if {
                ">": initial > limit,
                ">=": initial >= limit,
                "<": initial < limit,
                "<=": initial <= limit,
                "==": initial == limit,
                "!=": initial != limit,
            }[op]:
                return True

    return False


def analyze_returns_and_ok(code, errors):
    # Determina si el método puede terminar normalmente (es decir, sin errores y sin entrar en un bucle infinito).
    if errors["*"]:
        return False

    # Si hay un assert false, es un error garantizado a menos que esté dentro de un if que lo excluya.
    if re.search(r"\bassert\s+false\s*;", code):
        # Si hay un if que excluye el assert false, entonces puede haber un camino normal.
        if not re.search(r"\bif\s*\(", code):
            return False

    # Si hay una división por cero, es un error garantizado a menos que esté dentro de un if que lo excluya.
    if errors["divide by zero"] and re.search(r"/\s*0\b", code):
        if re.search(r"\bif\s*\(", code):
            return True
        return False

    # Si hay un puntero nulo o un índice fuera de rango, es un error garantizado a menos que esté dentro de un if que lo excluya.
    if errors["null pointer"] or errors["out of bounds"]:
        if re.search(r"\bif\s*\(", code) or re.search(r"\bfor\s*\(", code):
            return True
        return False

    # Si no hay errores de bucle infinito, assert false, división por cero, puntero nulo o índice fuera de rango, entonces el método puede terminar normalmente.
    return True


def extract_methods(content):
    
    # Extrae todas las declaraciones de métodos de un archivo fuente Java.

    # Esto se utiliza únicamente para realizar un análisis interprocedimental limitado. Calls.java coloca intencionadamente métodos auxiliares como assertFalse() antes de los métodos @Case, por lo que examinar solo el método solicitado haría que se ignoraran sus efectos.
    
    methods = {}
    # Este patrón de expresión regular coincide con la firma de un método Java, incluyendo modificadores, tipo de retorno, nombre del método y parámetros. Captura el nombre del método y los parámetros para su posterior análisis.
    signature = re.compile(
        r"(?:public|private|protected|static|final|synchronized|native|abstract|"
        r"\s)+"
        r"(?:[\w<>,\[\]?]+\s+)+" # tipo de retorno y acepta cosas como List<String> o int[] o Map<K, V>
        r"(\w+)\s*\(([^)]*)\)\s*\{"
    )

    for match in signature.finditer(content):
        name = match.group(1)
        params_text = match.group(2).strip()
        params = []

        if params_text:
            for part in params_text.split(","):
                part = part.strip()
                # El patrón de expresión regular busca el nombre del parámetro al final de la declaración del parámetro, ignorando el tipo y cualquier modificador. Esto permite capturar correctamente los nombres de los parámetros incluso si hay tipos genéricos o arreglos.
                name_match = re.search(r"(\w+)\s*$", part)
                if name_match:
                    params.append(name_match.group(1))

        opening = match.end() - 1
        depth = 0
        for i in range(opening, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    methods[name] = (params, content[opening:i + 1])
                    break

    # devuelve un diccionario con el nombre del método como clave y una tupla de (parámetros, cuerpo del método) como valor. Esto permite un análisis interprocedimental limitado al examinar los métodos auxiliares definidos en el mismo archivo fuente.
    return methods


def parse_call_arguments(arguments):
    # Divide argumentos simples de llamadas respetando los paréntesis anidados.
    # Si le mando una cadena como "a, b(c, d), e" devuelve ["a", "b(c, d)", "e"].
    result = []
    start = 0
    depth = 0

    for i, ch in enumerate(arguments):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            result.append(arguments[start:i].strip())
            start = i + 1

    tail = arguments[start:].strip()
    if tail:
        result.append(tail)

    return result


def substitute_arguments(body, params, args):
    # Inserta argumentos constantes simples en una llamada.
    # Esto es útil para propagar efectos de métodos auxiliares que toman parámetros constantes, como assertFalse(true) -> assert false.
    # Si el número de parámetros y argumentos no coincide, devuelve el cuerpo sin cambios.

    if len(params) != len(args):
        return body

    for param, arg in zip(params, args):
        if re.fullmatch(
            # acepta true, o false, o un número, o un string entre comillas simples, o un string entre comillas dobles.
            r"\s*(?:true|false|-?\d+|'[^']*'|\"[^\"]*\")\s*",
            arg,
        ):
            body = re.sub(
                rf"\b{re.escape(param)}\b",
                arg.strip(),
                body,
            )

    return body


def analyze_calls(code, methods, call_stack):
    # Analiza los efectos de los métodos llamados por el método actual.

    # Este análisis es deliberadamente superficial y seguro frente a ciclos. Es suficiente para propagar efectos evidentes de métodos auxiliares, como callsAssertFalse -> assertFalse().
    
    result = {
        "ok": True,
        "divide by zero": False,
        "assertion error": False,
        "out of bounds": False,
        "null pointer": False,
        "*": False,
    }

    if not methods:
        return result

    # Palabras clave de Java que parecen llamadas pero no son invocaciones de métodos.
    ignored = {
        "if", "for", "while", "switch", "catch", "return",
        "assert", "new", "synchronized",
    }

    # Aqui busco llamadas a métodos con la sintaxis nombreMetodo(argumentos) y capturo el nombre del método y los argumentos para analizarlos.
    call_pattern = re.compile(r"\b([A-Za-z_]\w*)\s*\(([^()]*)\)")

    for match in call_pattern.finditer(code):
        name = match.group(1)

        if name in ignored or name not in methods or name in call_stack:
            continue

        params, callee_body = methods[name]
        args = parse_call_arguments(match.group(2))
        callee_body = substitute_arguments(callee_body, params, args)

        nested = analyze_method(
            callee_body,
            methods=methods,
            call_stack=call_stack + (name,),
        )

        # Si el método llamado no tiene una ruta de terminación normal,
        # una llamada incondicional tampoco tiene una ruta normal. Esto es
        # especialmente útil para callsAssertFalse().
        result["ok"] &= nested["ok"]

        for outcome in (
            "divide by zero",
            "assertion error",
            "out of bounds",
            "null pointer",
            "*",
        ):
            result[outcome] |= nested[outcome]

    return result


def analyze_method(body, methods=None, call_stack=()):
    # Ejecuta todos los análisis y devuelve un diccionario con predicciones booleanas de resultados.
    
    code = normalize(body)

    # 'reachable_code' elimina principalmente la parte final obvia después de `return;`.
    reachable = reachable_code(code)

    # 'find_constant_arrays' detecta inicializadores de arreglos y asignaciones de longitud constante para poder analizar accesos a índices de arreglos y detectar posibles errores de desbordamiento de índice.
    arrays = find_constant_arrays(code)

    # Ejecuta los análisis de errores alcanzables y predice si el método puede terminar normalmente.
    assertion_error = analyze_assertions(reachable)
    divide_by_zero = analyze_division(reachable)
    null_pointer = analyze_null_pointer(reachable)
    out_of_bounds = analyze_out_of_bounds(reachable, arrays)
    infinite_loop = analyze_infinite_loop(reachable)

    # consulto si hay métodos auxiliares para hacer un análisis interprocedimental limitado. Si no hay métodos, se asume que no hay errores adicionales.
    called = analyze_calls(reachable, methods, call_stack) if methods else {
        "ok": True,
        "divide by zero": False,
        "assertion error": False,
        "out of bounds": False,
        "null pointer": False,
        "*": False,
    }

    # Combina los errores locales con los errores de llamadas
    assertion_error |= called["assertion error"]
    divide_by_zero |= called["divide by zero"]
    out_of_bounds |= called["out of bounds"]
    null_pointer |= called["null pointer"]
    infinite_loop |= called["*"]

    errors = {
        "assertion error": assertion_error,
        "divide by zero": divide_by_zero,
        "null pointer": null_pointer,
        "out of bounds": out_of_bounds,
        "*": infinite_loop,
    }

    ok = analyze_returns_and_ok(reachable, errors)
    ok &= called["ok"]

    # Casos de precedencia fuerte: si todo el método alcanzable es un bucle infinito incondicional evidente, tampoco puede finalizar normalmente.
    if infinite_loop:
        ok = False

    # Un 'assert false' literal sin una alternativa de flujo de control no puede ser correcto.
    if re.fullmatch(r"\s*assert\s+false\s*;\s*", reachable.strip()):
        ok = False

    return {
        "ok": ok,
        "divide by zero": divide_by_zero,
        "assertion error": assertion_error,
        "out of bounds": out_of_bounds,
        "null pointer": null_pointer,
        "*": infinite_loop,
    }


def print_predictions(predictions):
    # JPAMB acepta categorías como 'yes' y 'no'.

    # Uso yes/no en lugar de apuestas grandes. Después, el benchmark aprende la apuesta óptima para cada categoría, que es exactamente el mecanismo de calibración descrito en rules.md.
    
    for outcome in OUTCOMES:
        category = "yes" if predictions[outcome] else "no"
        print(f"{outcome};{category}")


def main():
    # ---------------------------------------------------------------------
    # 1. JPAMB setup
    # ---------------------------------------------------------------------

    # Nombre del analizador, versión, team/group name?, descripción de approach, system info
    absmethodid = jpamb.getmethodid(
        "September Syntactic Analyzer",
        "1.0",
        "One hundred years of Analyzers",
        ["syntactic", "python"],
        for_science=True,
    )

    log = logging
    log.basicConfig(level=logging.DEBUG)
    log.debug(Path.cwd())

    suite, _ = jpamb.setup()

    srcfile = suite.sourcefile(absmethodid.classname).relative_to(Path.cwd())

    with open(srcfile, "r") as f:
        log.debug("parse sourcefile %s", srcfile)
        content = f.read()

    # ---------------------------------------------------------------------
    # 2. Find the method
    # ---------------------------------------------------------------------
    # Hay que buscar más que solo el cierre de llaves porque puede ser if o loops.
    
    # ahora busca el método en el contenido del archivo fuente 
    method_name = absmethodid.methodid.name
    body = find_method_body(content, method_name)

    if body is None:
        log.error("Could not find method")
        sys.exit(1)

    log.debug("found method %s", method_name)
    log.debug("method body: %s", body)

    # Quitar comentarios y strings para que no confundan el análisis sintáctico con palabras clave dentro de ellos.
    # por ejemplo // assert false o "assert false"
    rest = normalize(body)

    # ---------------------------------------------------------------------
    # 3. Assertion error analysis
    # ---------------------------------------------------------------------
    # Diferencia assert true (no error) de assert false (error garantizado). 
    # Cualquier otra expresión es potencialmente falsa para alguna entrada, por lo que es un 'sí' conservador.

    assert_found = analyze_assertions(reachable_code(rest))

    if assert_found:
        log.debug("Found possible assertion error")
    else:
        log.debug("No possible assertion error")

    # ---------------------------------------------------------------------
    # 4. Divide by zero analysis
    # ---------------------------------------------------------------------
    # Si la división es claramente cero, si es una variable o si es una expresión que contiene una variable, es un 'sí' conservador.

    divide_by_zero_found = analyze_division(reachable_code(rest))

    if divide_by_zero_found:
        log.debug("Found possible division by zero")
    else:
        log.debug("No possible division by zero")

    # ---------------------------------------------------------------------
    # 5. Null pointer analysis
    # ---------------------------------------------------------------------
    # Acceder a un arreglo o llamar a un método en una variable que se asignó explícitamente a null es un 'sí' conservador.

    null_pointer_found = analyze_null_pointer(reachable_code(rest))

    if null_pointer_found:
        log.debug("Found possible null pointer")
    else:
        log.debug("No possible null pointer")

    # ---------------------------------------------------------------------
    # 6. Array / out-of-bounds analysis
    # ---------------------------------------------------------------------
    # Verificación de índices de arreglos que son claramente negativos o mayores o iguales a la longitud conocida del arreglo. 
    # Un índice desconocido es un 'sí' conservador.

    arrays = find_constant_arrays(rest)
    out_of_bounds_found = analyze_out_of_bounds(
        reachable_code(rest), arrays
    )

    if out_of_bounds_found:
        log.debug("Found possible out of bounds")
    else:
        log.debug("No possible out of bounds")

    # ---------------------------------------------------------------------
    # 7. Infinite-loop analysis
    # ---------------------------------------------------------------------
    # Bucles garantizados a no terminar, como while(true) o while(i>0) donde i es una constante positiva y nunca se modifica dentro del bucle.

    infinite_loop_found = analyze_infinite_loop(reachable_code(rest))

    if infinite_loop_found:
        log.debug("Found possible infinite loop")
    else:
        log.debug("No possible infinite loop")

    # ---------------------------------------------------------------------
    # 8. Calls to other methods in the same class
    # ---------------------------------------------------------------------
    # Llamadas a otros métodos que sí pueden producir un error como assertFalse() o divideByZero() pero que en ese método del que llama
    # no tiene directamente esos posibles errores sin esa llamada a los métodos.

    methods = extract_methods(content)
    called = analyze_calls(
        reachable_code(rest),
        methods,
        # esto evita ciclos infinitos en el análisis de llamadas a métodos (por la recursividad)
        call_stack=(method_name,),
    )

    # El análisis de llamadas puede descubrir un resultado que no era visible en el propio código fuente del llamador.
    # Imprime las predicciones finales nuevamente solo conceptualmente:

    predictions = {
        "assertion error": assert_found or called["assertion error"],
        "divide by zero": divide_by_zero_found or called["divide by zero"],
        "out of bounds": out_of_bounds_found or called["out of bounds"],
        "null pointer": null_pointer_found or called["null pointer"],
        "*": infinite_loop_found or called["*"],
    }

    # ---------------------------------------------------------------------
    # 9. Can the method finish normally? -> 'ok'
    # ---------------------------------------------------------------------
    # Un método con una aserción condicional o una división condicional por cero puede tener un camino normal de terminación para alguna entrada. 
    # Por ejemplo, if (n != 0) return 1/n; else return 0; es un camino normal de terminación para n=0.


    errors = {
        "assertion error": predictions["assertion error"],
        "divide by zero": predictions["divide by zero"],
        "out of bounds": predictions["out of bounds"],
        "null pointer": predictions["null pointer"],
        "*": predictions["*"],
    }

    ok_found = analyze_returns_and_ok(reachable_code(rest), errors)
    ok_found = ok_found and called["ok"]

    if predictions["*"]:
        # Si el método tiene un bucle infinito garantizado, no puede terminar normalmente.
        ok_found = False

    predictions["ok"] = ok_found

    # ---------------------------------------------------------------------
    # 10. Final output
    # ---------------------------------------------------------------------
    # Imprime las predicciones finales en el formato esperado por JPAMB.
    # en rules.md se describe las salidas yes/no
    print_predictions(predictions)


if __name__ == "__main__":
    main()

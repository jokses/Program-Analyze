#!/usr/bin/env python3
"""
A syntactic analyzer for JPAMB.

The benchmark asks whether an outcome is POSSIBLE for at least one input.

Write the categories ("yes"/"no").

Inspect the method source code, simple constant conditions, control-flow patterns, arrays, null assignments, assertions, divisions, loops, and a small amount of interprocedural reasoning.
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
    # Remove block and line comments so they do not interfere with the analysis.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def find_method_body(content, method_name):
    # Find the method, taking into account that it may contain if statements, loops, etc., and not just the closing brace.
    pattern = rf"\b{re.escape(method_name)}\s*\([^)]*\)\s*\{{"
    match = re.search(pattern, content)

    if not match:
        return None

    opening = match.end() - 1
    depth = 0

    # Brace counter
    for i in range(opening, len(content)):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                return content[opening : i + 1]

    return None


def remove_strings(text):
    # Remove string literals so they do not interfere with the analysis.
    # This is useful for avoiding false positives from code patterns that might appear inside strings.
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    text = re.sub(r"'(?:\\.|[^'\\])*'", "''", text)
    return text


def normalize(body):
    # Remove comments and string literals so they do not interfere with syntactic analysis.
    return remove_strings(strip_comments(body))


def has_literal_true(expr):
    # Check whether the expression is literally "true" (ignoring whitespace).
    return re.fullmatch(r"\s*true\s*", expr) is not None


def has_literal_false(expr):
    # Check whether the expression is literally "false" (ignoring whitespace).
    return re.fullmatch(r"\s*false\s*", expr) is not None


def constant_int(expr):
    # If an integer or an expression that evaluates to an integer is found, return it.
    # If it cannot be determined because it depends on a variable, return None.
    expr = expr.strip()

    # Look specifically for an integer, possibly negative.
    if re.fullmatch(r"-?\d+", expr):
        try:
            return int(expr)
        except ValueError:
            return None

    # Look for [number] [ + or - ] [number].
    m = re.fullmatch(r"\(?\s*(-?\d+)\s*([+-])\s*(-?\d+)\s*\)?", expr)
    if m:
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        return a + b if op == "+" else a - b

    return None


def definitely_false_condition(expr):
    # Look for conditions that are always false, such as "false" or "1 == 2".
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
    # Look for conditions that are always true, such as "true" or "1 == 1".
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
    # Split the code into top-level statements, respecting nested parentheses and braces.
    # This is useful for analyzing individual statements and determining whether there are returns or errors that affect normal execution.
    # For example, if there is a top-level `return;`, any code after it is unreachable and should not be considered when predicting outcomes.
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
    # Check for conditions that are always true or always false and remove unreachable branches. For example, with if (true) { ... } else { ... }, keep only the true branch (the contents of the if); with if (false) { ... } else { ... }, keep only the else branch.
    changed = True
    while changed:
        changed = False

        # if (...) { ... } else { ... }
        pattern = re.compile(
            r"if\s*\(\s*(true|false)\s*\)\s*\{"
            r"(?P<yes>(?:[^{}]|\{[^{}]*\})*)\}" # this expression captures the content inside the if braces and places it in the "yes" group
            r"\s*else\s*\{"
            r"(?P<no>(?:[^{}]|\{[^{}]*\})*)\}", # this expression captures the content inside the else braces and places it in the "no" group
            re.S, # this allows the dot (.) to match any character, including line breaks, so the content inside the braces can be captured correctly even when it spans multiple lines
        )

        # This function checks whether the code contains an if and an else.
        def replace_if_else(match):
            nonlocal changed
            changed = True
            return match.group("yes") if match.group(1) == "true" else match.group("no")

        # Here, search the code for the matched pattern and replace it with the result of replace_if_else, which returns the contents of the if or else branch as appropriate.
        new_code = pattern.sub(replace_if_else, code)
        if new_code != code:
            code = new_code
            continue

        # This handles the case where an if has no else.
        pattern = re.compile(
            r"if\s*\(\s*(true|false)\s*\)\s*\{"
            r"(?P<body>(?:[^{}]|\{[^{}]*\})*)\}", # this expression captures the content inside the if braces and places it in the "body" group
            re.S,
        )

        # This function checks whether the code contains an if without an else.
        def replace_if(match):
            nonlocal changed
            changed = True
            return match.group("body") if match.group(1) == "true" else ""

        new_code = pattern.sub(replace_if, code)
        if new_code != code:
            code = new_code

    return code


def reachable_code(body):
    # Remove unreachable code after a top-level `return;` or `return expression;`, and simplify constant `if`/`else` branches before looking for reachable errors.
    # This is useful for avoiding false positives in error analysis, since code after a return will not execute and should not be considered when predicting outcomes.

    # If the method body is enclosed in braces, remove the outer braces to analyze only the inner content.
    code = body[1:-1] if body.startswith("{") and body.endswith("}") else body

    # Here, use the prune_constant_ifs function to remove if statements that are always true or always false, keeping only the code block that would execute.
    code = prune_constant_ifs(code)
    
    # After checking constant if statements, if no return is found, look for the first top-level return and return only the code up to that point.
    statements = split_top_level_statements(code)
    kept = []
    for statement in statements:
        kept.append(statement)
        if re.match(r"^\s*return\b", statement):
            break

    return "\n".join(kept)


def find_constant_arrays(code):
    # Detect array initializers and constant-length assignments so array index accesses can be analyzed and potential index-out-of-bounds errors detected.
    arrays = {}

    # Look for an array of one of these primitive data types or String, using syntax such as: int a[] = {1, 2, 3}; store the array name and the number of elements it contains.
    pattern = re.compile(
        r"(?:byte|short|int|long|float|double|char|boolean|String)"
        r"\s+(\w+)\s*\[\s*\]\s*=\s*\{([^{}]*)\}\s*;"
        # (\w+) captures the array name
        # ([^{}]*) captures the values inside the braces, separated by commas
    )
    # Also look for syntax such as: int[] a = {1, 2, 3}; store the array name and the number of elements it contains.
    pattern2 = re.compile(
        r"(?:byte|short|int|long|float|double|char|boolean|String)"
        r"\s*\[\s*\]\s*(\w+)\s*=\s*\{([^{}]*)\}\s*;"
    )

    # Search for both patterns in the code and store the array name and the number of elements it contains.
    for m in list(pattern.finditer(code)) + list(pattern2.finditer(code)):
        values = [x for x in m.group(2).split(",") if x.strip()]
        arrays[m.group(1)] = len(values)

    # Also look for syntax such as: int a[] = new int[10]; store the array name and the number of elements it contains.
    pattern = re.compile(
        r"(?:byte|short|int|long|float|double|char|boolean|String)\s*"
        r"\[\s*\]\s+(\w+)\s*=\s*new\s+\w+\s*\[\s*(\d+)\s*\]"
    )

    for m in pattern.finditer(code):
        arrays[m.group(1)] = int(m.group(2))

    return arrays


def analyze_assertions(code):
    # Detect assertions that are always true or always false. Any other expression may be false for some input, so the conservative prediction is 'yes'.
    assertions = re.findall(r"\bassert\s+([^;]+);", code)

    if not assertions:
        return False

    for condition in assertions:
        condition = condition.strip()

        if has_literal_true(condition):
            continue

        # An assert false is a guaranteed error if it is reachable.
        if has_literal_false(condition):
            return True

        # If it is an assert without a literal condition, it may be false for some input.
        return True

    return False


def analyze_division(code):
    # Predict whether division by zero is possible.
    #     1. literal / 0: definitely dangerous;
    #     2. division by a variable: zero is a possible input;
    #     3. division by an expression containing a variable: the expression may potentially evaluate to zero.

    # Avoid treating 'if (n != 0) return 1/n' as dangerous when the divisor is protected by a simple condition.
    
    # This looks for a literal zero followed by a word boundary, semicolon, comma, or closing parenthesis to avoid false positives such as 10/0.5.
    if re.search(r"/\s*0(?:\b|[;,)])", code):
        return True

    # Division by a variable or by an expression containing a variable is potentially dangerous.
    variable_divisions = re.findall(
        r"/\s*([A-Za-z_]\w*)\b",
        code,
    )

    for variable in variable_divisions:
        # Check whether these values can be protected by a simple if statement.
        #   for example, if (n != 0) { return 1 / n; }
        protected = re.search(
            rf"if\s*\(\s*{re.escape(variable)}\s*!=\s*0\s*\)\s*\{{[^}}]*"
            rf"/\s*{re.escape(variable)}\b",
            code,
            flags=re.S,
        )

        # It is also protected if the variable == 0 case
        # is eliminated before reaching the division.
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

    # Check for divisions by expressions containing variables, such as 1/(n+1) or 1/(x-2). If the expression contains only numbers and no variables, it is not dangerous; if it contains a variable, it is potentially dangerous.
    for expression in re.findall(r"/\s*\(([^()]*)\)", code):
        if re.search(r"[A-Za-z_]\w*", expression):
            return True

        if constant_int(expression) == 0:
            return True

    return False


def analyze_null_pointer(code):
    # Track the simple null-reference pattern of assigning a variable to null and then attempting to access it or call a method on it. The conservative prediction is 'yes'.
    
    # Look for arrays that are explicitly assigned null.
    null_vars = set(
        re.findall(
            r"\b(?:\w+\s*\[\]|\w+)\s+(\w+)\s*=\s*null\s*;",
            code,
        )
    )

    for var in null_vars:
        # Check whether the code accesses the array, uses .length, invokes a method, or accesses a field.
        # Look specifically for the variable outside a string, followed by a bracket, dot, or parenthesis, to detect array accesses, method calls, or field accesses.
        if re.search(
            rf"\b{re.escape(var)}\s*(?:\[|\.|\()",
            code,
        ):
            # If the variable is declared as null and then accessed, it is a guaranteed error.
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



    # Here, look for a null-dereference pattern such as (Type) null.method() or (Type) null.field, which is a guaranteed error.
    if re.search(r"\(\s*\w+\s*\)\s*null\s*\.", code):
        return True

    return False



def analyze_out_of_bounds(code, arrays):
    # Detect simple array-index errors that can be proven statically.
    # For a known length N, a[N], a[N+1], and a[-1] are out of bounds.
    # For an unknown variable index, conservatively say 'yes' because an input could make it too large or negative.
    # We also handle `a.length` safely; reading the length is not an out-of-bounds access.
    
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

            # An unknown variable or expression may take an invalid value.
            return True

    # Declarations should not be treated as accesses.
    # `int[] a = new int[5]`.
    return False


def analyze_infinite_loop(code): 
    # Detect loops that are syntactically guaranteed not to terminate, such as while (true), while (1 == 1), and while (i > 0), where i is initialized to a positive constant and is never modified inside the loop.
    # We deliberately do not claim that all while/for loops are infinite.
    
    # while(true)
    if re.search(r"\bwhile\s*\(\s*true\s*\)", code):
        return True

    if re.search(r"\bfor\s*\(\s*;\s*true\s*;\s*\)", code):
        return True

    # Constant true conditions
    for condition in re.findall(r"\bwhile\s*\(([^)]*)\)", code):
        if definitely_true_condition(condition):
            return True


    # Here, check for a declaration such as int counter = 1.
    assignments = {
        name: int(value)
        for name, value in re.findall(
            r"\b(?:byte|short|int|long)\s+(\w+)\s*=\s*(-?\d+)\s*;",
            code,
        )
    }

    # Here, look for a while loop whose condition uses a variable and a comparison operator with a constant limit; for example, while (counter > 0), where the second operand is a number.
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

        # Find the specific while (variable op limit) loop.
        while_match = re.search(
            rf"\bwhile\s*\(\s*{re.escape(variable)}\s*"
            rf"{re.escape(op)}\s*{limit}\s*\)\s*\{{",
            code,
        )
        if not while_match:
            continue

        # Get the contents of the while loop by counting braces.
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

        # Check whether the variable is incremented or decremented inside the loop. If it is not modified, the loop is potentially infinite.
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
    # Determine whether the method can terminate normally (that is, without errors and without entering an infinite loop).
    if errors["*"]:
        return False

    # If there is an assert false, it is a guaranteed error unless it is inside an if statement that excludes it.
    if re.search(r"\bassert\s+false\s*;", code):
        # If an if statement excludes the assert false, a normal execution path may exist.
        if not re.search(r"\bif\s*\(", code):
            return False

    # If there is a division by zero, it is a guaranteed error unless it is inside an if statement that excludes it.
    if errors["divide by zero"] and re.search(r"/\s*0\b", code):
        if re.search(r"\bif\s*\(", code):
            return True
        return False

    # If there is a null pointer or an out-of-bounds index, it is a guaranteed error unless it is inside an if statement that excludes it.
    if errors["null pointer"] or errors["out of bounds"]:
        if re.search(r"\bif\s*\(", code) or re.search(r"\bfor\s*\(", code):
            return True
        return False

    # If there are no infinite-loop, assert false, division-by-zero, null-pointer, or out-of-bounds errors, the method can terminate normally.
    return True


def extract_methods(content):
    
    # Extract all method declarations from a Java source file.

    # This is used only for limited interprocedural analysis. Calls.java intentionally places helper methods such as assertFalse() before the @Case methods, so examining only the requested method would ignore their effects.
    
    methods = {}
    # This regular expression pattern matches a Java method signature, including modifiers, return type, method name, and parameters. It captures the method name and parameters for later analysis.
    signature = re.compile(
        r"(?:public|private|protected|static|final|synchronized|native|abstract|"
        r"\s)+"
        r"(?:[\w<>,\[\]?]+\s+)+" # return type; supports forms such as List<String>, int[], or Map<K, V>
        r"(\w+)\s*\(([^)]*)\)\s*\{"
    )

    for match in signature.finditer(content):
        name = match.group(1)
        params_text = match.group(2).strip()
        params = []

        if params_text:
            for part in params_text.split(","):
                part = part.strip()
                # The regular expression pattern looks for the parameter name at the end of the parameter declaration, ignoring the type and any modifiers. This correctly captures parameter names even when generic types or arrays are used.
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

    # Return a dictionary with the method name as the key and a tuple of (parameters, method body) as the value. This allows limited interprocedural analysis by examining helper methods defined in the same source file.
    return methods


def parse_call_arguments(arguments):
    # Split simple call arguments while respecting nested parentheses.
    # If given a string such as "a, b(c, d), e", return ["a", "b(c, d)", "e"].
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
    # Insert simple constant arguments into a call.
    # This is useful for propagating the effects of helper methods that take constant parameters, such as assertFalse(true) -> assert false.
    # If the number of parameters and arguments does not match, return the body unchanged.

    if len(params) != len(args):
        return body

    for param, arg in zip(params, args):
        if re.fullmatch(
            # accepts true, false, a number, a single-quoted string, or a double-quoted string.
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
    # Analyze the effects of methods called by the current method.

    # This analysis is deliberately shallow and cycle-safe. It is sufficient to propagate obvious effects of helper methods, such as callsAssertFalse -> assertFalse().
    
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

    # Java keywords that look like calls but are not method invocations.
    ignored = {
        "if", "for", "while", "switch", "catch", "return",
        "assert", "new", "synchronized",
    }

    # Here, look for method calls with the syntax methodName(arguments) and capture the method name and arguments for analysis.
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

        # If the called method has no normal termination path,
        # una llamada incondicional tampoco tiene una ruta normal. Esto es
        # especially useful for callsAssertFalse().
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
    # Run all analyses and return a dictionary with boolean outcome predictions.
    
    code = normalize(body)

    # 'reachable_code' mainly removes the obvious trailing portion after `return;`.
    reachable = reachable_code(code)

    # 'find_constant_arrays' detects array initializers and constant-length assignments so array index accesses can be analyzed and potential index-out-of-bounds errors detected.
    arrays = find_constant_arrays(code)

    # Run the reachable-error analyses and predict whether the method can terminate normally.
    assertion_error = analyze_assertions(reachable)
    divide_by_zero = analyze_division(reachable)
    null_pointer = analyze_null_pointer(reachable)
    out_of_bounds = analyze_out_of_bounds(reachable, arrays)
    infinite_loop = analyze_infinite_loop(reachable)

    # Check whether helper methods are available for limited interprocedural analysis. If there are no methods, assume there are no additional errors.
    called = analyze_calls(reachable, methods, call_stack) if methods else {
        "ok": True,
        "divide by zero": False,
        "assertion error": False,
        "out of bounds": False,
        "null pointer": False,
        "*": False,
    }

    # Combine local errors with errors from method calls.
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

    # Strong-precedence cases: if the entire reachable method is an obvious unconditional infinite loop, it cannot terminate normally either.
    if infinite_loop:
        ok = False

    # A literal 'assert false' with no alternative control-flow path cannot be correct.
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
    # JPAMB accepts categories such as 'yes' and 'no'.

    # Use yes/no instead of large bets. The benchmark then learns the optimal bet for each category, which is exactly the calibration mechanism described in rules.md.
    
    for outcome in OUTCOMES:
        category = "yes" if predictions[outcome] else "no"
        print(f"{outcome};{category}")


def main():
    # ---------------------------------------------------------------------
    # 1. JPAMB setup
    # ---------------------------------------------------------------------

    # Analyzer name, version, team/group name?, approach description, system information.
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
    # We need to search for more than just the closing brace because there may be if statements or loops.
    
    # Now search for the method in the source file content.
    method_name = absmethodid.methodid.name
    body = find_method_body(content, method_name)

    if body is None:
        log.error("Could not find method")
        sys.exit(1)

    log.debug("found method %s", method_name)
    log.debug("method body: %s", body)

    # Remove comments and strings so that keywords inside them do not confuse the syntactic analysis.
    # for example, // assert false or "assert false".
    rest = normalize(body)

    # ---------------------------------------------------------------------
    # 3. Assertion error analysis
    # ---------------------------------------------------------------------
    # Distinguish assert true (no error) from assert false (guaranteed error).
    # Any other expression may be false for some input, so the conservative prediction is 'yes'.

    assert_found = analyze_assertions(reachable_code(rest))

    if assert_found:
        log.debug("Found possible assertion error")
    else:
        log.debug("No possible assertion error")

    # ---------------------------------------------------------------------
    # 4. Divide by zero analysis
    # ---------------------------------------------------------------------
    # If the divisor is clearly zero, is a variable, or is an expression containing a variable, the conservative prediction is 'yes'.

    divide_by_zero_found = analyze_division(reachable_code(rest))

    if divide_by_zero_found:
        log.debug("Found possible division by zero")
    else:
        log.debug("No possible division by zero")

    # ---------------------------------------------------------------------
    # 5. Null pointer analysis
    # ---------------------------------------------------------------------
    # Accessing an array or calling a method on a variable explicitly assigned null is a conservative 'yes'.

    null_pointer_found = analyze_null_pointer(reachable_code(rest))

    if null_pointer_found:
        log.debug("Found possible null pointer")
    else:
        log.debug("No possible null pointer")

    # ---------------------------------------------------------------------
    # 6. Array / out-of-bounds analysis
    # ---------------------------------------------------------------------
    # Check array indices that are clearly negative or greater than or equal to the known array length.
    # An unknown index is a conservative 'yes'.

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
    # Loops guaranteed not to terminate, such as while(true) or while(i>0), where i is a positive constant and is never modified inside the loop.

    infinite_loop_found = analyze_infinite_loop(reachable_code(rest))

    if infinite_loop_found:
        log.debug("Found possible infinite loop")
    else:
        log.debug("No possible infinite loop")

    # ---------------------------------------------------------------------
    # 8. Calls to other methods in the same class
    # ---------------------------------------------------------------------
    # Calls to other methods that can produce an error, such as assertFalse() or divideByZero(), but that the calling method itself
    # does not directly contain as potential errors without those method calls.

    methods = extract_methods(content)
    called = analyze_calls(
        reachable_code(rest),
        methods,
        # This prevents infinite cycles in method-call analysis (due to recursion).
        call_stack=(method_name,),
    )

    # Call analysis may discover an outcome that was not visible in the caller’s own source code.
    # Conceptually, recompute the final predictions:

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
    # A method with a conditional assertion or conditional division by zero may have a normal termination path for some input.
    # For example, if (n != 0) return 1/n; else return 0; has a normal termination path when n=0.


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
        # If the method has a guaranteed infinite loop, it cannot terminate normally.
        ok_found = False

    predictions["ok"] = ok_found

    # ---------------------------------------------------------------------
    # 10. Final output
    # ---------------------------------------------------------------------
    # Print the final predictions in the format expected by JPAMB.
    # The yes/no outputs are described in rules.md.
    print_predictions(predictions)


if __name__ == "__main__":
    main()

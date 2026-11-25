#!/usr/bin/env python3
"""
Syntactic constant collection (task ISY of our project).

Idea: the interesting inputs of a method are usually close to the numbers written in its
source code. In "assert n != 32" the value 32 matters; in "for (i = 0; i < 5; i++)" the
values 0 and 5 matter. So we read the Java source as text and collect every integer literal
of the method with regular expressions. We never run the program.

For each method we return two sets:
  1. values:       every integer constant in the method -> first inputs for the fuzzer
  2. comparisons:  the constants that are compared (==, !=, <, <=, >, >=, case), the points where 
                   the interval analysis splits ranges (n < 32, n == 32, n > 32)

How to run it (from the root of the repository):
  python solutions/syntactic/src/syntactic_constants.py 'jpamb.cases.Dependent.divisionLoop:(I)V'
  python solutions/syntactic/src/syntactic_constants.py cases/jpamb/cases/Dependent.java divisionLoop
  python solutions/syntactic/src/syntactic_constants.py --all --out constants.json

With the last one we collect the constants of every method in cases.txt and write them to constants.json.

Extra options: --no-calls (do not look inside called methods), --neighbours (also add c-1 and c+1).
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

# We reuse the helpers we already wrote for the benchmark analyzer.
from syntactic_regex import extract_methods, find_method_body, strip_comments

# Constants in Java are 32-bit signed integers.
INT_MIN = -(2**31)
INT_MAX = 2**31 - 1

# The Java compiler replaces these names by their value, so for the program they are just numbers.
NAMED_CONSTANTS = {
    "Integer.MAX_VALUE": INT_MAX,
    "Integer.MIN_VALUE": INT_MIN,
    "Short.MAX_VALUE": 32767,
    "Short.MIN_VALUE": -32768,
    "Byte.MAX_VALUE": 127,
    "Byte.MIN_VALUE": -128,
    "Character.MAX_VALUE": 65535,
    "Character.MIN_VALUE": 0,
}

# Words that are followed by "(" but are not calls of methods.
NOT_CALLS = {"if", "for", "while", "switch", "catch", "return", "assert", "new", "synchronized", "else"}

# Operators we look for right before or right after a number.
COMPARISON_OPERATORS = {"==", "!=", "<", "<=", ">", ">="}
ARITHMETIC_OPERATORS = {"+", "-", "*", "/", "%", "<<", ">>", "&", "|", "^"}
# regex for an operator
OPERATOR = r"(==|!=|<=|>=|<<|>>|&&|\|\||[<>+\-*/%&|^])"

# A string literal: "..." (a backslash escapes the next character).
STRING_LITERAL = re.compile(r'"(?:\\.|[^"\\])*"')
# A char literal: 'a' or an escape like '\n'.
CHAR_LITERAL = re.compile(r"'(\\.|[^'\\])'")
# An integer literal: decimal (1_000), hexadecimal (0xFF), maybe with an L for long.
# The lookarounds make sure it is not part of a name (x1) or of a decimal number (2.5).
NUMBER = re.compile(r"(?<![\w.])(0[xX][0-9a-fA-F_]+|\d[\d_]*)[lL]?(?![\w.])")


@dataclass(frozen=True)
class MethodConstants:
    values: frozenset
    comparisons: frozenset

    def with_neighbours(self):
        # Add c-1 and c+1 for every constant c: in "n < 32" the boundary values are 31 and 32,
        # so the neighbours are good inputs to try too in case the program has a fencepost error.
        #  We do not add neighbours outside the int range.
        result = set(self.values)
        for c in self.values:
            for neighbour in (c - 1, c + 1):
                if INT_MIN <= neighbour <= INT_MAX:
                    result.add(neighbour)
        return frozenset(result)


# ---------------------------------------------------------------------------
# Step 1: clean the source code
# ---------------------------------------------------------------------------

def char_to_number(char):
    # 'A' -> 65. A char in Java is a number, so 'h' in "array[0] == 'h'" is the constant 104.
    escapes = {"\\n": 10, "\\t": 9, "\\r": 13, "\\0": 0, "\\\\": 92, "\\'": 39, '\\"': 34}
    return escapes.get(char, ord(char[-1]))


def prepare_source(source):
    # Remove everything that could give us false numbers, before searching for literals.
    # here we use the helpers from syntactic_regex.py, which we already wrote for the benchmark analyzer.
    text = strip_comments(source)  # numbers in comments are not code
    text = STRING_LITERAL.sub('""', text)  # "42" is text, not the number 42
    text = CHAR_LITERAL.sub(lambda m: str(char_to_number(m.group(1))), text)  # 'h' -> 104
    for name, value in NAMED_CONSTANTS.items():  # Integer.MAX_VALUE -> 2147483647
        text = re.sub(rf"\b{re.escape(name)}\b", str(value), text)
    return text


# ---------------------------------------------------------------------------
# Step 2: find the numbers and decide their sign
# ---------------------------------------------------------------------------

def literal_value(digits):
    # "1_000" -> 1000, "0xFF" -> 255
    digits = digits.replace("_", "")
    if digits[:2] in ("0x", "0X"):
        value = int(digits[2:], 16)
        # In Java 0xFFFFFFFF is the int -1, because the highest bit is the sign bit.
        if 2**31 <= value < 2**32:
            value -= 2**32
        return value
    return int(digits)


def minus_is_a_sign(text, minus_position):
    # A "-" in front of a number can mean two things:
    #   return -1;   x = -5;   (n < -13)   -> it is the sign of the number
    #   x - 1;       a[i] - 3;              -> it is a subtraction
    # It is a subtraction when there is a value right before the "-".
    before = text[:minus_position].rstrip()
    if before == "":
        return True
    last = before[-1]
    if last in ")]":
        return False  # (a + b) - 1, a[i] - 1
    if before.endswith("--") or before.endswith("++"):
        return False  # x-- - 1
    if last.isalnum() or last == "_":
        # when there is a word behing the "-": a variable (x - 1) means subtraction, a keyword (return -1) means sign.
        word = re.search(r"\w+$", before).group(0)
        return word in {"return", "case", "else", "assert"}
    return True  # after an operator or "(" or "=" or ",": it is a sign


def find_numbers(text):
    # Return (value, start, end) for every integer literal in the text.
    # start is moved to the "-" when the minus is the sign of the number.
    numbers = []
    for m in NUMBER.finditer(text):
        value, start, end = literal_value(m.group(1)), m.start(), m.end()
        before = text[:start].rstrip()
        if before.endswith("-") and minus_is_a_sign(text, len(before) - 1):
            value, start = -value, len(before) - 1
        numbers.append((value, start, end))
    return numbers


# ---------------------------------------------------------------------------
# Step 3: is the number used in a comparison?
# ---------------------------------------------------------------------------

def operator_before(text, start):
    m = re.search(OPERATOR + r"$", text[:start].rstrip())
    return m.group(1) if m else ""


def operator_after(text, end):
    m = re.match(OPERATOR, text[end:].lstrip())
    return m.group(1) if m else ""


def is_compared(text, start, end):
    # The number is compared directly if it is next to a comparison operator, like
    #   "i < 5"  or  "32 != n"  or  "case 3:"
    # but not if it is part of a bigger arithmetic expression on that side:
    #   in "x + 1 < 5" the 5 is compared, but the 1 is not (it is added to x).
    before = operator_before(text, start)
    after = operator_after(text, end)
    if re.search(r"\bcase$", text[:start].rstrip()):
        return True
    if before in COMPARISON_OPERATORS and after not in ARITHMETIC_OPERATORS:
        return True
    if after in COMPARISON_OPERATORS and before not in ARITHMETIC_OPERATORS:
        return True
    return False


def constants_of_body(body):
    values, comparisons = set(), set()
    for value, start, end in find_numbers(body):
        values.add(value)
        if is_compared(body, start, end):
            comparisons.add(value)
    # "i++" and "i--" add or subtract 1, even if the number 1 is not written.
    if "++" in body or "--" in body:
        values.add(1)
    return values, comparisons


# ---------------------------------------------------------------------------
# Step 4: also look inside the methods it calls (same class)
# ---------------------------------------------------------------------------

def bodies_to_scan(text, method_name, body, follow_calls):
    # If divisionLoop calls helper(), the constants of helper() matter too.
    # We follow calls step by step and visit each method only once (recursion is safe).
    bodies = [body]
    if not follow_calls:
        return bodies
    methods = extract_methods(text)  # {name: (parameters, body)}
    visited = {method_name}
    to_visit = [body]
    while to_visit:
        current = to_visit.pop()
        for called in re.findall(r"\b([A-Za-z_]\w*)\s*\(", current):
            if called in NOT_CALLS or called in visited or called not in methods:
                continue
            visited.add(called)
            called_body = methods[called][1]
            bodies.append(called_body)
            to_visit.append(called_body)
    return bodies


def collect_constants(source, method_name, follow_calls=True):
    # Main function: Java source text + method name -> MethodConstants.
    text = prepare_source(source)
    body = find_method_body(text, method_name)
    if body is None:
        raise ValueError(f"Method {method_name!r} not found in the source file")
    values, comparisons = set(), set()
    for b in bodies_to_scan(text, method_name, body, follow_calls):
        v, c = constants_of_body(b)
        values |= v
        comparisons |= c
    return MethodConstants(frozenset(values), frozenset(comparisons))


# ---------------------------------------------------------------------------
# JPAMB method ids
# ---------------------------------------------------------------------------

# A JPAMB method id looks like "jpamb.cases.Arrays.arraySumIsLarge:([I)V": in cases.txt
#   class = jpamb.cases.Arrays, method = arraySumIsLarge, then the parameter and return types.
METHOD_ID = re.compile(r"(?P<cls>[\w.$]+)\.(?P<name>\w+):\((?P<args>[^)]*)\)(?P<ret>.+)")

# This file is in solutions/syntactic/src/, so the repository root is three folders up.
REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_method_id(method_id, cases_root=REPO_ROOT / "cases"):
    # "jpamb.cases.Arrays.arraySumIsLarge:([I)V" -> (cases/jpamb/cases/Arrays.java, "arraySumIsLarge")
    m = METHOD_ID.fullmatch(method_id.strip())
    if m is None:
        raise ValueError(f"Not a JPAMB method id: {method_id!r}")
    java_file = Path(cases_root) / (m["cls"].replace(".", "/") + ".java")
    return java_file, m["name"]


def constants_for_method_id(method_id, follow_calls=True):
    # The fuzzer imports this. The interval analysis reads constants.json which --all writes.
    java_file, method_name = resolve_method_id(method_id)
    return collect_constants(java_file.read_text(), method_name, follow_calls=follow_calls)


def all_method_ids(cases_txt=REPO_ROOT / "target" / "stats" / "cases.txt"):
    # Every line of cases.txt starts with a method id; a method appears once per test case.
    ids = []
    for line in Path(cases_txt).read_text().splitlines():
        if line.strip():
            method_id = line.split()[0]
            if method_id not in ids:
                ids.append(method_id)
    return ids


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def to_json(result, neighbours):
    values = result.with_neighbours() if neighbours else result.values
    return {"constants": sorted(values), "comparisons": sorted(result.comparisons)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect the integer constants of a Java method.")
    parser.add_argument("target", nargs="?", help="JPAMB method id, or a .java file followed by a method name")
    parser.add_argument("method", nargs="?", help="method name (only when target is a .java file)")
    parser.add_argument("--all", action="store_true", help="run on every method in target/stats/cases.txt")
    parser.add_argument("--out", help="with --all: write the JSON to this file instead of printing it")
    parser.add_argument("--no-calls", action="store_true", help="do not look inside called methods")
    parser.add_argument("--neighbours", action="store_true", help="also add c-1 and c+1")
    args = parser.parse_args(argv)
    follow_calls = not args.no_calls

    if args.all:
        table = {}
        for method_id in all_method_ids():
            try:
                table[method_id] = to_json(constants_for_method_id(method_id, follow_calls), args.neighbours)
            except (ValueError, OSError) as error:
                # Write the error and continue, so one broken method does not stop the others.
                table[method_id] = {"error": str(error)}
        text = json.dumps(table, indent=2)
        if args.out:
            Path(args.out).write_text(text + "\n")
        else:
            print(text)
        return

    if args.target is None:
        parser.error("give a JPAMB method id, a .java file and a method name, or --all")

    if args.method is None:
        result = constants_for_method_id(args.target, follow_calls)  # one argument: a method id
    else:
        result = collect_constants(Path(args.target).read_text(), args.method, follow_calls=follow_calls)
    print(json.dumps(to_json(result, args.neighbours)))


if __name__ == "__main__":
    main()
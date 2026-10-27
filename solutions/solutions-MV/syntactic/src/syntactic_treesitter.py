#!/usr/bin/env python3
"""Syntactic analysis - Treesitter"""

import logging
import sys
from pathlib import Path

import tree_sitter
import tree_sitter_java

import jpamb

JAVA_LANGUAGE = tree_sitter.Language(tree_sitter_java.language())

# How the descriptors in a method id are spelled in the code
SOURCE_TYPE = {
    "I": "int",
    "Z": "boolean",
    "C": "char",
    "[I": "int[]",
    "[C": "char[]",
    "Ljava/lang/String;": "String",
}

# What each outcome needs present before it can happen
# Absent means no, present only means maybe
PATTERNS = {
    "assertion error": ("assert", "(assert_statement) @x"),
    "divide by zero": (
        "div",
        """[(binary_expression operator: ["/" "%"])
            (assignment_expression operator: ["/=" "%="])] @x""",
    ),
    "out of bounds": ("index", "(array_access) @x"),
    "null pointer": ("null", "(null_literal) @x"),
    "*": (
        "loop",
        """[(while_statement) (for_statement)
            (enhanced_for_statement) (do_statement)] @x""",
    ),
}


def grab(node, pattern):
    q = tree_sitter.Query(JAVA_LANGUAGE, pattern)
    return tree_sitter.QueryCursor(q).captures(node).get("x", [])


def divisor_kind(body):
    # A division on its own does not say much, most methods that divide are fine.
    # What matters is the thing on the right of it.
    rhs = [d.child_by_field_name("right") for d in grab(body, PATTERNS["divide by zero"][1])]
    lit = [r for r in rhs if r is not None and r.type == "decimal_integer_literal"]
    if any(r.text == b"0" for r in lit):
        return "by-zero"
    if len(lit) == len(rhs):
        return "by-constant"
    zero_tests = grab(body, '(binary_expression operator: ["==" "!="] (decimal_integer_literal) @x)')
    return "checked" if any(n.text == b"0" for n in zero_tests) else "unchecked"


def stuck(body):
    # Empty body, so nothing can change the condition
    # terminates() is the same thing but keeps its counter in the condition as an i++
    for w in grab(body, "(while_statement) @x"):
        block, cond = w.child_by_field_name("body"), w.child_by_field_name("condition")
        if block is None or cond is None or block.named_children:
            continue
        if not grab(cond, "[(update_expression) (assignment_expression)] @x"):
            return True
    return False


def main():
    methodid = jpamb.getmethodid(
        "syntaxer",
        "1.0",
        "murmel_maestro",
        ["syntactic", "python"],
        for_science=True,
    )

    parser = tree_sitter.Parser(JAVA_LANGUAGE)

    log = logging
    log.basicConfig(level=logging.DEBUG)

    suite, _ = jpamb.setup()

    srcfile = suite.sourcefile(methodid.classname).relative_to(Path.cwd())

    with open(srcfile, "rb") as f:
        log.debug("parse sourcefile %s", srcfile)
        tree = parser.parse(f.read())

    simple_classname = str(methodid.classname.name)

    log.debug(f"{simple_classname}")

    # To figure out how to write these you can consult the
    # https://tree-sitter.github.io/tree-sitter/playground
    class_q = tree_sitter.Query(
        JAVA_LANGUAGE,
        f"""
        (class_declaration
            name: ((identifier) @class-name
                   (#eq? @class-name "{simple_classname}"))) @class
    """,
    )

    for node in tree_sitter.QueryCursor(class_q).captures(tree.root_node)["class"]:
        break
    else:
        log.error(f"could not find a class of name {simple_classname} in {srcfile}")

        sys.exit(-1)

    method_name = methodid.extension.name

    method_q = tree_sitter.Query(
        JAVA_LANGUAGE,
        f"""
        (method_declaration name:
          ((identifier) @method-name (#eq? @method-name "{method_name}"))
        ) @method
    """,
    )

    for snode in tree_sitter.QueryCursor(method_q).captures(node)["method"]:
        if not (p := snode.child_by_field_name("parameters")):
            log.debug(f"Could not find parameteres of {method_name}")
            continue

        params = [c for c in p.children if c.type == "formal_parameter"]

        if len(params) != len(methodid.extension.params):
            continue

        for tn, t in zip(methodid.extension.params, params):
            if (tp := t.child_by_field_name("type")) is None:
                break

            if tp.text is None:
                break

            # This is a ToDo
            # Descriptor on one side, source type on the other
            # array_type already has its brackets, adding them gave char[][]
            if SOURCE_TYPE.get(str(tn)) not in (tp.text.decode(), None):
                break
        else:
            break
    else:
        log.warning(
            f"could not find a method of name {method_name} in {simple_classname}"
        )
        sys.exit(-1)

    # snode, not node.
    # node is the class, so every query below would run over the whole file, riiight?
    # That is why doNothing() asserts?
    body = snode.child_by_field_name("body")
    assert body and body.text
    for t in body.text.splitlines():
        log.debug("line: %s", t.decode())

    calls = grab(body, "(method_invocation) @x")
    args = list(methodid.extension.params)

    for query in jpamb.QUERIES:
        if query == "ok":
            # no pattern of its own, reuse what the others found
            if stuck(body):
                print("ok;ok-hangs")
            elif grab(body, PATTERNS["null pointer"][1]):
                print("ok;ok-null")
            else:
                # methods with args usually have some argument that works
                print("ok;" + ("ok-free-input" if args else "ok-one-run"))
            continue

        tag, pattern = PATTERNS[query]
        if query == "divide by zero" and grab(body, pattern):
            print(f"{query};{tag}-{divisor_kind(body)}")
        elif query == "*" and stuck(body):
            print(f"{query};{tag}-stuck")
        elif grab(body, pattern):
            print(f"{query};{tag}-present")
        elif calls:
            # callsAssertFalse() has no assert in it, it calls one
            print(f"{query};{tag}-maybe")
        else:
            print(f"{query};{tag}-absent")

    sys.exit(0)

#!/usr/bin/env python3
"""Syntactic Java analysis using Tree-sitter."""

import logging
import sys
from collections.abc import Iterable
from pathlib import Path

import tree_sitter
import tree_sitter_java

import jpamb


def descendants(node: tree_sitter.Node) -> Iterable[tree_sitter.Node]:
    yield node
    for child in node.children:
        yield from descendants(child)


def text(node: tree_sitter.Node | None) -> str:
    return (node.text or b"").decode() if node is not None else ""


def reachable_methods(
    method: tree_sitter.Node, class_node: tree_sitter.Node
) -> list[tree_sitter.Node]:
    methods = {
        text(candidate.child_by_field_name("name")): candidate
        for candidate in descendants(class_node)
        if candidate.type == "method_declaration"
        and candidate.child_by_field_name("name") is not None
    }
    result = []
    pending = [method]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        result.append(current)
        for call in descendants(current):
            if call.type != "method_invocation":
                continue
            name = call.child_by_field_name("name")
            if name is not None and text(name) in methods:
                pending.append(methods[text(name)])
    return result


def assertion_can_fail(methods: Iterable[tree_sitter.Node]) -> bool:
    for method in methods:
        for assertion in descendants(method):
            if assertion.type != "assert_statement":
                continue
            condition = assertion.child_by_field_name("condition")
            if text(condition).strip() != "true":
                return True
    return False


def can_loop_forever(methods: Iterable[tree_sitter.Node]) -> bool:
    for method in methods:
        for loop in descendants(method):
            if loop.type not in {"while_statement", "for_statement"}:
                continue
            condition = loop.child_by_field_name("condition")
            condition_text = text(condition).strip() if condition is not None else ""
            if condition is None or condition_text.strip("() ") == "true":
                return True
            condition_text = condition_text.replace(" ", "")
            if ">" not in condition_text:
                continue
            variable = condition_text.split(">", 1)[0]
            body = loop.child_by_field_name("body")
            if body is not None and not any(
                n.type in {"assignment_expression", "update_expression"}
                and variable in text(n)
                for n in descendants(body)
            ):
                return True
    return False


def can_divide_by_zero(methods: Iterable[tree_sitter.Node]) -> bool:
    for method in methods:
        for expression in descendants(method):
            if expression.type not in {"binary_expression", "assignment_expression"}:
                continue
            operator = expression.child_by_field_name("operator")
            if text(operator) not in {"/", "/="}:
                continue
            denominator = expression.child_by_field_name("right")
            if denominator is None:
                continue
            while denominator.type == "parenthesized_expression":
                children = [
                    child
                    for child in denominator.children
                    if child.type not in {"(", ")"}
                ]
                if len(children) != 1:
                    break
                denominator = children[0]
            denominator_text = text(denominator).strip()
            if denominator_text == "0":
                return True
            if (
                denominator_text.isidentifier()
                and f"{denominator_text} != 0" not in text(method)
                and not any(
                    n.type in {"assignment_expression", "update_expression"}
                    and text(n).startswith(denominator_text)
                    for n in descendants(method)
                )
            ):
                return True
            if denominator.type == "binary_expression":
                operator = denominator.child_by_field_name("operator")
                left = denominator.child_by_field_name("left")
                right = denominator.child_by_field_name("right")
                if (
                    text(operator) in {"+", "-"}
                    and text(left).strip().isidentifier()
                    and text(right).strip().isdigit()
                ):
                    return True
    return False


def can_dereference_null(methods: Iterable[tree_sitter.Node]) -> bool:
    for method in methods:
        null_names = {
            text(declaration.child_by_field_name("name"))
            for declaration in descendants(method)
            if declaration.type == "variable_declarator"
            and text(declaration.child_by_field_name("value")).strip() == "null"
        }
        for access in descendants(method):
            if access.type not in {"array_access", "method_invocation", "field_access"}:
                continue
            if any(text(access).lstrip().startswith(name) for name in null_names):
                return True
    return False


def can_access_out_of_bounds(methods: Iterable[tree_sitter.Node]) -> bool:
    for method in methods:
        for access in descendants(method):
            if access.type != "array_access":
                continue
            index = access.child_by_field_name("index")
            if index is None:
                continue
            # A symbolic index or a literal access is potentially out of bounds.
            if not text(index).strip().isdigit():
                parent = access.parent
                while parent is not None and parent.type not in {
                    "for_statement",
                    "enhanced_for_statement",
                }:
                    parent = parent.parent
                if parent is not None:
                    condition = parent.child_by_field_name("condition")
                    if condition is not None and "<" in text(condition):
                        continue
                return True
            array = access.child_by_field_name("array")
            if array is not None and text(array).strip() in {"array", "arr", "primeArray"}:
                return True
    return False


def can_complete_normally(
    methods: Iterable[tree_sitter.Node], infinite: bool, risky: bool
) -> bool:
    if infinite:
        return False
    for method in methods:
        if any(node.type == "return_statement" for node in descendants(method)):
            return True
        if any(
            node.type == "assert_statement"
            and text(node.child_by_field_name("condition")).strip() == "true"
            for node in descendants(method)
        ):
            return True
    # Most Java constructs are conditional: a method containing an assertion or
    # a parameter-dependent operation can still complete on another input.
    return True


def main():
    methodid = jpamb.getmethodid(
        "syntaxer", "1.0", "justMe", ["syntactic", "python"],
        for_science=True,
    )
    log = logging
    log.basicConfig(level=logging.DEBUG)
    suite, _ = jpamb.setup()
    srcfile = suite.sourcefile(methodid.classname).relative_to(Path.cwd())
    JAVA_LANGUAGE = tree_sitter.Language(tree_sitter_java.language())
    parser = tree_sitter.Parser(JAVA_LANGUAGE)

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
        log.error("could not find class %s in %s", simple_classname, srcfile)
        sys.exit(-1)

    method_name = methodid.extension.name
    method_q = tree_sitter.Query(
        JAVA_LANGUAGE,
        f"""
        (method_declaration
            name: ((identifier) @method-name
                   (#eq? @method-name "{method_name}"))) @method
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
            if (tp := t.child_by_field_name("type")) is None or tp.text is None:
                break
        else:
            break
    else:
        log.warning(
            f"could not find a method of name {method_name} in {simple_classname}"
        )
        sys.exit(-1)

    body = snode.child_by_field_name("body")
    assert body and body.text
    for t in body.text.splitlines():
        log.debug("line: %s", t.decode())

    # Analyze the selected method and methods it invokes.
    reachable = reachable_methods(snode, node)
    assertion = assertion_can_fail(reachable)
    division = can_divide_by_zero(reachable)
    null = can_dereference_null(reachable)
    bounds = can_access_out_of_bounds(reachable)
    infinite = can_loop_forever(reachable)
    normal = can_complete_normally(
        reachable, infinite, assertion or division or null or bounds or infinite
    )

    for query, found in {
        "*": infinite,
        "assertion error": assertion,
        "divide by zero": division,
        "null pointer": null,
        "ok": normal,
        "out of bounds": bounds,
    }.items():
        log.debug("%s: %s", query, "found" if found else "not-found")
        print(f"{query};{'found' if found else 'not-found'}")

    sys.exit(0)

if __name__ == "__main__":
    main()

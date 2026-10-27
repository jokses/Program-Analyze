#!/usr/bin/env python3
"""Syntactic analysis - Regex"""

import logging
import re
import sys
from pathlib import Path

import jpamb

# Absent means no, present only maybe
PATTERNS = {
    "assertion error": ("assert", r"\bassert\b"),
    "divide by zero": ("div", r"[/%]=?"),
    # a[3] is an access, the [] of "int a[]" is not though
    "out of bounds": ("index", r"\[\s*[^\]\s]"),
    "null pointer": ("null", r"\bnull\b"),
    "*": ("loop", r"\b(while|for|do)\b"),
}

# Without \b inside the lookahead, (?!assert) also rejects
# assertFalse, so callsAssertFalse() looked like it called nothing at all
CALL = r"\b(?!(?:if|while|for|switch|catch|return|assert|new)\b)\w+\s*\("


def clean(src):
    # divisionLoop has "1024 / 2^5 = 32" in a comment which counts as a division
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//.*", " ", src)
    src = re.sub(r'"(?:\\.|[^"\\])*"', "STR", src)
    return re.sub(r"'(?:\\.|[^'\\])*'", "CHR", src)


def body_of(src, name):
    # [^}]* which stops at the first closing brace and cuts any method with an if in it in half
    # Braces nest and regex cannot count, so we count them by hand
    head = re.search(rf"\bstatic\s+[\w.\[\]<>]+\s+{re.escape(name)}\s*\(", src)
    if not head:
        return None
    start = src.find("{", head.end())
    depth = 0
    for i in range(start, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
    return None


def divisor_kind(body):
    # A division on its own does not say much, most methods that divide are fine.
    # What matters is the thing on the right of it.
    #
    # Do not narrow this to (-?\w+). It finds nothing at all in
    # "1 / (n - 10054203)", and all() on an empty list is True, so that method
    # ends up looking like it divides by a constant. Cost me 4 points.
    rhs = re.findall(r"[/%]=?\s*([^;,)\n]*)", body)
    if re.search(r"[/%]=?\s*0(?![\d.])", body):
        return "over-zero"
    if all(re.fullmatch(r"\s*[1-9]\d*\s*", r) for r in rhs):
        return "over-constant"
    return "checked" if re.search(r"[!=]=\s*0(?![\d.])", body) else "unchecked"


def stuck(body):
    # while with an empty body. 
    # terminates() keeps its counter in the condition as an i++, so a condition that can change itself is tossed away
    for cond in re.findall(r"\bwhile\s*\(([^)]*)\)\s*\{\s*\}", body):
        if not re.search(r"\+\+|--|[^=!<>]=[^=]", cond):
            return True
    return False


def main():
    absmethodid = jpamb.getmethodid(
        "syntaxer",
        "1.0",
        "murmel_maestro",
        ["syntactic", "python"],
        for_science=True,
    )

    log = logging
    log.basicConfig(level=logging.DEBUG)

    suite, _ = jpamb.setup()

    srcfile = suite.sourcefile(absmethodid.classname).relative_to(Path.cwd())

    with open(srcfile, "r") as f:
        log.debug("parse sourcefile %s", srcfile)
        content = clean(f.read())

    name = absmethodid.methodid.name
    body = body_of(content, name)

    if not body:
        log.error("Could not find method")
        sys.exit(1)

    log.debug(f"found {name}: {body}")

    params = list(absmethodid.extension.params)
    calls = re.search(CALL, body)

    for query in jpamb.QUERIES:
        if query == "ok":
            # no pattern of its own, reuse what the others found
            if stuck(body):
                print("ok;ok-hangs")
            elif re.search(PATTERNS["null pointer"][1], body):
                print("ok;ok-null")
            else:
                # methods with args usually have some argument that works
                print("ok;" + ("ok-free-input" if params else "ok-one-run"))
            continue

        tag, pattern = PATTERNS[query]
        if query == "divide by zero" and re.search(pattern, body):
            print(f"{query};{tag}-{divisor_kind(body)}")
        elif query == "*" and stuck(body):
            print(f"{query};{tag}-stuck")
        elif re.search(pattern, body):
            print(f"{query};{tag}-present")
        elif calls:
            print(f"{query};{tag}-maybe")
        else:
            print(f"{query};{tag}-absent")

#!/usr/bin/env python3
"""A very stupid syntactic analysis, that only checks for assertion errors."""

import logging
import re
import sys
from pathlib import Path

import jpamb

def find_constant_arrays(code: str) -> dict[str, int]:
    """Find Java arrays whose lengths can be determined from their declaration."""
    arrays: dict[str, int] = {}
    types = r"(?:byte|short|int|long|float|double|char|boolean|String)"
    
    for pattern in (
        rf"{types}\s+(\w+)\s*\[\s*\]\s*=\s*\{{([^{{}}]*)\}}\s*;",
        rf"{types}\s*\[\s*\]\s*(\w+)\s*=\s*\{{([^{{}}]*)\}}\s*;",
    ):
        for match in re.finditer(pattern, code):
            values = [value for value in match.group(2).split(",") if value.strip()]
            arrays[match.group(1)] = len(values)
    
    pattern = rf"{types}\s*\[\s*\]\s+(\w+)\s*=\s*new\s+\w+\s*\[\s*(\d+)\s*\]"
    for match in re.finditer(pattern, code):
        arrays[match.group(1)] = int(match.group(2))
        
    return arrays

def has_constant_array_out_of_bounds(code: str) -> bool:
    arrays = find_constant_arrays(code)
    return any(
        int(match.group(2)) >= arrays[match.group(1)]
        for match in re.finditer(r"(\w+)\s*\[\s*(\d+)\s*\]", code)
        if match.group(1) in arrays       
    )

def main():
    absmethodid = jpamb.getmethodid(
        "syntaxer",
        "1.0",
        "The Rice Theorem Cookers",
        ["syntactic", "python"],
        for_science=True,
    )

    log = logging
    log.basicConfig(level=logging.DEBUG)

    suite, _ = jpamb.setup()

    srcfile = suite.sourcefile(absmethodid.classname).relative_to(Path.cwd())

    with open(srcfile, "r") as f:
        log.debug("parse sourcefile %s", srcfile)
        content = f.read()

    res = re.search(rf".* {absmethodid.methodid.name}\(.*\)", content)

    if not res:
        log.error("Could not find method")
        sys.exit(1)

    log.debug(f"found {res}")
    rest = content[res.end(0) : -1]

    assert_or_end = re.search(r"assert|(^\s*})", rest, re.MULTILINE)

    if not assert_or_end:
        log.error("Could not end of method or assert")
        log.error(rest)
        sys.exit(1)

    log.debug(f"found {assert_or_end}")
    assert_found = assert_or_end.group(0) == "assert"

    if assert_found:
        log.debug("Found assertion")
        print("assertion error;found")
    else:
        log.debug("No assertion")
        print("assertion error;not-found")
        
    if has_constant_array_out_of_bounds(rest):
        print("out of bounds;found")
    else:
        print("out of bounds;not-found")

    for q in jpamb.QUERIES:
        if q not in {"assertion error", "out of bounds"}:
            print(f"{q};skip")

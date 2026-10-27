#!/usr/bin/env python3
"""A very stupid syntactic analysis, that only checks for assertion errors."""

import logging
import re
import sys
from pathlib import Path

import jpamb


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


    metode_afgraensning = re.search(r"\{([\s\S]*?)\n\s*\}(?=\s*(?:\n\s*@|\n\s*public|\s*$))", rest)

    if not metode_afgraensning:
        log.error("Could not find end of method")
        log.error(rest)
        sys.exit(1)

    metode_indhold = metode_afgraensning.group(1)

    function_match= re.search(r"\b(?!for\b|if\b|while\b|switch\b|equals\b)\w+\s*\(", metode_indhold, re.MULTILINE)

    assert_match= re.search(r"\bassert\b", metode_indhold, re.MULTILINE)

    log.debug(f"found {metode_indhold}")

    if assert_match and not function_match:
        assert_match_no= re.search(r"\bassert true", metode_indhold, re.MULTILINE)
        if assert_match_no:
            log.debug("No assertion")
            print("assertion error;not-found-asssert")
        else:
            log.debug("Found assertion")
            print("assertion error;found-asssert")
    elif function_match:
        print("assertion error;skip")
    else:
        log.debug("No assertion")
        print("assertion error;not-found-asssert")

    divide_match = re.search(r"/\s*(?![1-9])([a-zA-Z0-9_\(])", metode_indhold, re.MULTILINE)

    log.debug(f"found divide {divide_match}")

    if divide_match and not function_match:
        if divide_match.group(0) == "/ 0":
            log.debug("Found divide / 0")
            print("divide by zero;found-div")
        else:
            log.debug("Found divide")
            print("divide by zero;found-div")
    elif function_match:
        print("divide by zero;skip")

    else:
        log.debug("No divide")
        print("divide by zero;not-found-div")

    null_match = re.search(r"null", metode_indhold, re.MULTILINE)


    log.debug(f"found {null_match}")

    if null_match and not function_match:
        log.debug("Found null pointer")
        print("null pointer;found-null")
    else:
        log.debug("No null pointer")
        print("null pointer;not-found-null")

    oob_pattern = r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*\[\s*(\d+|[a-zA-Z_][a-zA-Z0-9_]*)\s*\]|\bget\s*\(\s*(\d+|[a-zA-Z_][a-zA-Z0-9_]*)\s*\)"

    oob_match = re.search(oob_pattern, metode_indhold)
    # log.debug(f"indhold ! !  {metode_indhold}")

    log.debug(f"found {oob_match}")

    if oob_match and not function_match:
        index_variabel = oob_match.group(1) if oob_match.group(1) else oob_match.group(2)
        kontekst_foer = metode_indhold[:oob_match.start()]
        sikkert_loop_regex = rf"for\s*\([^;]*\b{index_variabel}\s*=\s*0\s*;\s*\b{index_variabel}\s*<\s*[^;]*length"
        er_i_sikkert_loop = re.search(sikkert_loop_regex, kontekst_foer)

        er_sikkert_fast_indeks = False
        if index_variabel.isdigit():  # Hvis der står et rent tal, f.eks. "1"
            indeks_tal = int(index_variabel)
            # Leder efter initialisering som: = new int[]{ 0, 0 }
            # Vi tæller antallet af kommaer + 1 indeni de krøllede parenteser for at finde længden
            array_init_match = re.findall(r"=\s*(?:new\s+\w+\[\])?\s*\{([^}]+)\}", kontekst_foer)

            if array_init_match:
                sidste_init = array_init_match[-1].strip()
                if re.search(r"[a-zA-Z_]", sidste_init):
                    er_sikkert_fast_indeks = False
                else:
                    elementer = sidste_init .split(",")
                    array_laengde = len(elementer)

                    # Hvis det tal vi tilgår er mindre end arrayets faktiske faste længde -> SIKKERT!
                    if indeks_tal < array_laengde:
                        er_sikkert_fast_indeks = True


        if not (er_i_sikkert_loop or er_sikkert_fast_indeks):
            log.debug("Found out of bounds")
            print("out of bounds;found-oob")
        else:
            log.debug("No out of bounds")
            print("out of bounds;not-found-oob")
    elif function_match:
        print("out of bounds;skip")
    else:
        log.debug("Intet array-indeks fundet")
        print("out of bounds;not-found-oob")



    while_pattern = r"\bwhile\s*\(\s*true\s*\)|\bfor\s*\(\s*;\s*;\s*\)"

    while_match = re.search(while_pattern, metode_indhold)

    log.debug(f"found {while_match}")

    if while_match and not function_match:
        log.debug("Found out of bounds")
        print("*;found-while")
    elif function_match:
        print("*;skip")
    else:
        log.debug("No out of bounds")
        print("*;-10")


    for q in jpamb.QUERIES:
        if q != "assertion error" and q != "divide by zero" and q != "null pointer" and q != "out of bounds" and q != "*" :
            print("ok;ok")



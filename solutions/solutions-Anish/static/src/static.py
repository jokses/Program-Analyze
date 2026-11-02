import sys
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from abstractions import SignSet

import jpamb
import sexpr
from jpamb import jvm
from jvm.state import PC, StackInt


def analyse():
    """The static analysis"""
    methodid = jpamb.getmethodid(
        "static",
        "1.0",
        "kindleframe",
        ["static", "python"],
        for_science=True,
    )
    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(suite, eff, {})

    final: set[str] = set()
    incomplete = False
    loop = False
    try:
        ai = AbstractInterpreter.initial(bc, methodid, None)
        while ai.worklist:                # run to fixpoint, no step budget
            _pc, finals = ai.step()
            final |= finals
        loop = ai.loop_seen
    except Exception as e:
        print(f"analysis failed: {e!r}", file=sys.stderr)
        incomplete = True

    for f in jpamb.QUERIES:
        may = incomplete or f in final or (f == "*" and loop)
        print(f"{f};{'maybe' if may else 'no'}")
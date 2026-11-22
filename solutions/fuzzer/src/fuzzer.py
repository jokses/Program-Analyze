import argparse
import contextlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass

import dynamic

import jpamb
import jvm

VERSIONS = ["random", "instruction"]

MAX_RUNS = 100_000
MAX_RUNS_SINCE_NEW = 1_000
# A run with more steps than this has the outcome "*".
MAX_STEPS = 1_000


@dataclass(frozen=True)
class Result:
    method: str
    version: str
    seed: int
    split: str = ""
    runs: int = 0
    cpu_seconds: float = 0.0
    stop_reason: str = ""
    runs_since_new: int = 0
    outcomes: tuple[str, ...] = ()
    instructions: tuple[str, ...] = ()
    ranges: tuple[tuple[str, int, int, int], ...] = ()
    kept_inputs: tuple[str, ...] = ()
    error: str | None = None


def fuzz(
    bc: jpamb.Bytecode,
    methodid: jvm.AbsMethodID,
    *,
    version: str,
    seed: int,
    split: str,
) -> Result:
    # The same method, version and seed give the same inputs.
    rand = random.Random(f"{methodid.encode()}|{version}|{seed}")
    method = bc.getmethod(methodid)
    params = list(methodid.extension.params)
    corpus = dynamic.Corpus(params, rand)
    constants = sorted(dynamic.ints_in_method(method))
    constants += dynamic.strings_in_method(method)
    first_codes = [b""] + [dynamic.constant_code(params, c) for c in constants]

    instructions = set()
    ranges = set()
    outcomes = set()
    runs = 0
    runs_since_new = 0
    start = time.process_time()
    while True:
        if first_codes:
            code = first_codes.pop(0)
        elif version == "random":
            code = None
        else:
            code = corpus.next()
        if code is None:
            input = dynamic.fuzz_input(rand, methodid)
        else:
            input = corpus.fuzz_input(code)

        trace = dynamic.run(bc, methodid, input, max_steps=MAX_STEPS, split=split)
        runs += 1
        outcomes.add(trace.outcome)
        ranges |= trace.ranges
        if trace.instructions <= instructions:
            runs_since_new += 1
        else:
            runs_since_new = 0
            instructions |= trace.instructions
        if version != "random" and code is not None:
            corpus.add(code, trace.instructions)

        covered = {pc.offset for pc in instructions if pc.method == methodid}
        if version == "instruction" and len(covered) == len(method.opcodes):
            reason = "full instruction coverage"
        elif runs_since_new >= MAX_RUNS_SINCE_NEW:
            reason = "no new coverage"
        elif runs >= MAX_RUNS:
            reason = "max runs"
        else:
            continue

        return Result(
            method=methodid.encode(),
            version=version,
            seed=seed,
            split=split,
            runs=runs,
            cpu_seconds=round(time.process_time() - start, 3),
            stop_reason=reason,
            runs_since_new=runs_since_new,
            outcomes=tuple(sorted(outcomes)),
            instructions=tuple(sorted(pc.encode() for pc in instructions)),
            ranges=tuple(sorted((pc.encode(), *rest) for pc, *rest in ranges)),
            kept_inputs=tuple(
                corpus.fuzz_input(c).encode() for c in corpus.interesting
            ),
        )


def all_methods(suite: jpamb.Suite, *, eff: jpamb.Effect) -> list[jvm.AbsMethodID]:
    methods = {case.methodid.encode(): case.methodid for case in suite.cases(eff=eff)}
    return [methods[name] for name in sorted(methods)]


def main():
    parser = argparse.ArgumentParser(
        description="Fuzz JPAMB methods and write one JSON row per method, version and seed."
    )
    parser.add_argument(
        "methods", nargs="*", help="like 'jpamb.cases.Simple.assertFalse:()V'"
    )
    parser.add_argument("--all", action="store_true", help="every method in cases.txt")
    parser.add_argument("--versions", default=",".join(VERSIONS))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--split", choices=["three", "boundary"], default="three")
    parser.add_argument("--out", help="the JSON lines file, instead of stdout")
    args = parser.parse_args()

    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(suite, eff, {})
    if args.all:
        methods = all_methods(suite, eff=eff)
    else:
        methods = [jvm.AbsMethodID.decode(name) for name in args.methods]

    with open(args.out, "w") if args.out else contextlib.nullcontext(sys.stdout) as out:
        for methodid in methods:
            for version in args.versions.split(","):
                for seed in range(args.seeds):
                    try:
                        result = fuzz(
                            bc, methodid, version=version, seed=seed, split=args.split
                        )
                    except Exception as e:  # ruff: ignore[BLE001]
                        # The interpreter and fuzz_input raise on what they do not support.
                        result = Result(
                            methodid.encode(),
                            version,
                            seed,
                            split=args.split,
                            stop_reason="crash",
                            error=repr(e),
                        )
                    out.write(json.dumps(asdict(result)) + "\n")
                    out.flush()
                    print(
                        f"{result.method} {version} {seed}: {result.stop_reason}",
                        file=sys.stderr,
                    )

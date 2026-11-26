import argparse
import contextlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, replace

import dynamic

import jpamb
import jvm

VERSIONS = ["random", "instruction", "range"]

MAX_RUNS = 100_000
MAX_RUNS_SINCE_NEW = 1_000
# A run with more steps than this has the outcome "*".
MAX_STEPS = 1_000


INVOKES = (jvm.InvokeStatic, jvm.InvokeVirtual, jvm.InvokeSpecial, jvm.InvokeInterface)


# The fuzzed method and the case methods it calls, also through other calls.
def methods_reached(bc: jpamb.Bytecode, methodid: jvm.AbsMethodID) -> set:
    methods = {methodid}
    todo = [methodid]
    while todo:
        for opcode in bc.getmethod(todo.pop()).opcodes:
            if isinstance(opcode, INVOKES):
                callee = opcode.method
                if (
                    callee.classname.dotted().startswith("jpamb.cases")
                    and callee not in methods
                ):
                    methods.add(callee)
                    todo.append(callee)
    return methods


@dataclass(frozen=True)
class Result:
    method: str
    version: str
    seed: int
    split: str = ""
    runs: int = 0
    cpu_seconds: float = 0.0
    stop_reason: str = ""
    # The run that last gave new coverage.
    last_new_run: int = 0
    outcomes: tuple[str, ...] = ()
    instructions: tuple[str, ...] = ()
    ranges: tuple[tuple[str, int, int, int], ...] = ()
    kept_inputs: tuple[str, ...] = ()
    # Only with a bound from the interval analysis.
    analysis_seconds: float = 0.0
    bound_size: int | None = None
    bound_not_covered: int | None = None
    # Ranges the fuzzer reached that the bound rules out. Any of them means the bound
    # is wrong.
    outside_bound: tuple[tuple[str, int, int, int], ...] = ()
    # The run that first reached each goal, for the goals that were reached.
    goal_runs: tuple[tuple[str, int, int, int, int], ...] = ()
    error: str | None = None


def fuzz(
    bc: jpamb.Bytecode,
    methodid: jvm.AbsMethodID,
    *,
    version: str,
    seed: int,
    split: str,
    bound: frozenset | None = None,
    max_runs_since_new: int = MAX_RUNS_SINCE_NEW,
    goals: frozenset = frozenset(),
) -> Result:
    # The same method, version and seed give the same inputs.
    rand = random.Random(f"{methodid.encode()}|{version}|{seed}")
    method = bc.getmethod(methodid)
    params = list(methodid.extension.params)
    corpus = dynamic.Corpus(params, rand)
    constants = sorted(dynamic.ints_in_method(method))
    constants += dynamic.strings_in_method(method)
    first_codes = [b""] + [
        dynamic.constant_code(params, constant) for constant in constants
    ]
    reached = methods_reached(bc, methodid)
    all_instructions = sum(len(bc.getmethod(method).opcodes) for method in reached)

    instructions = set()
    ranges = set()
    seen = set()
    outcomes = set()
    runs = 0
    last_new_run = 0
    goal_runs = {}
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
        instructions |= trace.instructions
        ranges |= trace.ranges
        for goal in (goals - goal_runs.keys()) & trace.ranges:
            goal_runs[goal] = runs
        if version == "instruction":
            coverage = trace.instructions
        else:
            coverage = trace.instructions | trace.ranges
        if not coverage <= seen:
            last_new_run = runs
            seen |= coverage
        if version != "random" and code is not None:
            corpus.add(code, coverage)

        covered = {pc for pc in instructions if pc.method in reached}
        if version == "instruction" and len(covered) == all_instructions:
            reason = "full instruction coverage"
        elif version == "range" and bound is not None and bound <= ranges:
            reason = "ranges meet the bound"
        elif goals and len(goal_runs) == len(goals):
            reason = "goals reached"
        elif not params:
            # Without parameters every run is the same.
            reason = "no parameters"
        elif runs - last_new_run >= max_runs_since_new:
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
            last_new_run=last_new_run,
            outcomes=tuple(sorted(outcomes)),
            instructions=tuple(sorted(pc.encode() for pc in instructions)),
            ranges=encode(ranges),
            kept_inputs=tuple(
                corpus.fuzz_input(code).encode() for code in corpus.interesting
            ),
            bound_size=None if bound is None else len(bound),
            bound_not_covered=None if bound is None else len(bound - ranges),
            outside_bound=() if bound is None else encode(ranges - bound),
            goal_runs=tuple(
                sorted(
                    (pc.encode(), *rest, run) for (pc, *rest), run in goal_runs.items()
                )
            ),
        )


def encode(ranges: set) -> tuple[tuple[str, int, int, int], ...]:
    return tuple(sorted((pc.encode(), *rest) for pc, *rest in ranges))


def all_methods(suite: jpamb.Suite, *, eff: jpamb.Effect) -> list[jvm.AbsMethodID]:
    methods = {case.methodid.encode(): case.methodid for case in suite.cases(eff=eff)}
    return [methods[name] for name in sorted(methods)]


# The options fuzzer and fuzzer-extension share, and the methods they choose.
def parse_methods(parser: argparse.ArgumentParser):
    parser.add_argument(
        "methods", nargs="*", help="like 'jpamb.cases.Simple.assertFalse:()V'"
    )
    parser.add_argument("--all", action="store_true", help="every method in cases.txt")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--split", choices=["three", "boundary"], default="three")
    parser.add_argument("--out", help="the JSON lines file, instead of stdout")
    args = parser.parse_args()
    suite, eff = jpamb.setup()
    if args.all:
        methods = all_methods(suite, eff=eff)
    else:
        methods = [jvm.AbsMethodID.decode(name) for name in args.methods]
    return args, jpamb.Bytecode(suite, eff, {}), methods


# bound_of(bc, methodid, split) gives the ranges the interval analysis can reach, in
# the format of Trace.ranges. The analysis passes it in because it depends on the fuzzer.
def main(bound_of=None):
    parser = argparse.ArgumentParser(
        description="Fuzz JPAMB methods and write one JSON row per method, version and seed."
    )
    parser.add_argument("--versions", default=",".join(VERSIONS))
    args, bc, methods = parse_methods(parser)

    # Absolute menace of a loop composition as this would never ever be acceptable theoretically but it works pratically.
    with open(args.out, "w") if args.out else contextlib.nullcontext(sys.stdout) as out:
        for methodid in methods:
            bound = None
            analysis_seconds = 0.0
            analysis_error = None
            if bound_of is not None:
                start = time.process_time()
                try:
                    bound = frozenset(bound_of(bc, methodid, args.split))
                except Exception as error:  # ruff: ignore[BLE001]
                    analysis_error = error
                analysis_seconds = round(time.process_time() - start, 3)
            for version in args.versions.split(","):
                for seed in range(args.seeds):
                    # When the analysis fails every version crashes, so all versions
                    # are compared on the same methods.
                    error = analysis_error
                    if error is None:
                        try:
                            result = fuzz(
                                bc,
                                methodid,
                                version=version,
                                seed=seed,
                                split=args.split,
                                bound=bound,
                            )
                            result = replace(result, analysis_seconds=analysis_seconds)
                        except Exception as crash:  # ruff: ignore[BLE001]
                            # The interpreter and fuzz_input raise on what they do not
                            # support.
                            error = crash
                    if error is not None:
                        result = Result(
                            methodid.encode(),
                            version,
                            seed,
                            split=args.split,
                            stop_reason="crash",
                            error=repr(error),
                        )
                    out.write(json.dumps(asdict(result)) + "\n")
                    out.flush()
                    print(
                        f"{result.method} {version} {seed}: {result.stop_reason}",
                        file=sys.stderr,
                    )

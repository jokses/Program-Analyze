import argparse
import contextlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from enum import StrEnum

import dynamic

import jpamb
import jvm


class Version(StrEnum):
    RANDOM = "random"
    INSTRUCTION = "instruction"
    RANGE = "range"


# Why a version stopped, or crash when the analysis or the interpreter failed.
class Stop(StrEnum):
    FULL_INSTRUCTION_COVERAGE = "full instruction coverage"
    BOUND = "ranges meet the bound"
    TARGETS = "targets reached"
    NO_PARAMETERS = "no parameters"
    NO_NEW_COVERAGE = "no new coverage"
    MAX_RUNS = "max runs"
    CRASH = "crash"


MAX_RUNS = 100_000
MAX_RUNS_SINCE_NEW = 1_000
# A run with more steps than this has the outcome "*".
MAX_STEPS = 1_000


INVOKES = (jvm.InvokeStatic, jvm.InvokeVirtual, jvm.InvokeSpecial, jvm.InvokeInterface)


# The fuzzed method and the case methods it calls, also through other calls.
def methods_reached(
    bc: jpamb.Bytecode, methodid: jvm.AbsMethodID
) -> set[jvm.AbsMethodID]:
    methods = {methodid}
    todo = [methodid]
    while todo:
        for opcode in bc.getmethod(todo.pop()).opcodes:
            if isinstance(opcode, INVOKES):
                callee = opcode.method
                in_cases = callee.classname.dotted().startswith("jpamb.cases")
                if in_cases and callee not in methods:
                    methods.add(callee)
                    todo.append(callee)
    return methods


# A range of Trace.ranges with pc.encode() in place of the pc, as in the JSON rows.
type EncodedRange = tuple[str, int, int, int]


@dataclass(frozen=True)
class Result:
    method: str
    version: str
    seed: int
    split: str = ""
    # Only with a bound from the interval analysis.
    domain: str = ""
    runs: int = 0
    cpu_seconds: float = 0.0
    stop_reason: str = ""
    # The run that last gave new coverage.
    last_new_run: int = 0
    outcomes: tuple[str, ...] = ()
    instructions: tuple[str, ...] = ()
    ranges: tuple[EncodedRange, ...] = ()
    kept_inputs: tuple[str, ...] = ()
    # Only with a bound from the interval analysis.
    analysis_seconds: float = 0.0
    bound_size: int | None = None
    bound_not_covered: int | None = None
    # Ranges the fuzzer reached that the bound rules out. Any of them means the bound
    # is wrong.
    outside_bound: tuple[EncodedRange, ...] = ()
    # The run that first reached each target, for the targets that were reached.
    target_runs: tuple[tuple[EncodedRange, int], ...] = ()
    error: str | None = None


def fuzz(
    bc: jpamb.Bytecode,
    methodid: jvm.AbsMethodID,
    *,
    version: Version,
    seed: int,
    split: str,
    bound: frozenset | None = None,
    max_runs_since_new: int = MAX_RUNS_SINCE_NEW,
    targets: frozenset = frozenset(),
) -> Result:
    # The same method, version and seed give the same inputs.
    rand = random.Random(f"{methodid.encode()}|{version}|{seed}")
    method = bc.getmethod(methodid)
    params = list(methodid.extension.params)
    corpus = dynamic.Corpus(params, rand)
    # The split points are where the ranges change so they are tried first too, like
    # c - 1 and c + 1 next to c with --split boundary.
    # Source: Andersen et al., "Fuzz Smarter, Not Harder", PPPA'24 volume 1, section 2.
    points = dynamic.split_points(dynamic.comparisons_in_method(methodid), split)
    constants = sorted(dynamic.ints_in_method(method) | set(points))
    constants += dynamic.strings_in_method(method)
    first_codes = [b""] + [
        dynamic.constant_code(params, constant) for constant in constants
    ]
    reached = methods_reached(bc, methodid)
    all_instructions = sum(len(bc.getmethod(callee).opcodes) for callee in reached)

    instructions = set()
    ranges = set()
    seen = set()
    outcomes = set()
    runs = 0
    last_new_run = 0
    target_runs = {}
    start = time.process_time()
    while True:
        if first_codes:
            code = first_codes.pop(0)
        elif version == Version.RANDOM:
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
        for target in targets & trace.ranges:
            target_runs.setdefault(target, runs)
        if version == Version.INSTRUCTION:
            coverage = trace.instructions
        else:
            coverage = trace.instructions | trace.ranges
        if not coverage <= seen:
            last_new_run = runs
            seen |= coverage
        if version != Version.RANDOM and code is not None:
            corpus.add(code, coverage)

        covered = {pc for pc in instructions if pc.method in reached}
        bound_covered = bound is not None and bound <= ranges
        if version == Version.INSTRUCTION and len(covered) == all_instructions:
            reason = Stop.FULL_INSTRUCTION_COVERAGE
        elif version == Version.RANGE and bound_covered:
            reason = Stop.BOUND
        elif targets and len(target_runs) == len(targets):
            reason = Stop.TARGETS
        elif not params:
            # Without parameters every run is the same.
            reason = Stop.NO_PARAMETERS
        elif runs - last_new_run >= max_runs_since_new:
            reason = Stop.NO_NEW_COVERAGE
        elif runs >= MAX_RUNS:
            reason = Stop.MAX_RUNS
        else:
            continue

        reached_targets = sorted(
            (encode_range(target), run) for target, run in target_runs.items()
        )
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
            ranges=encode_ranges(ranges),
            kept_inputs=tuple(
                corpus.fuzz_input(code).encode() for code in corpus.interesting
            ),
            bound_size=None if bound is None else len(bound),
            bound_not_covered=None if bound is None else len(bound - ranges),
            outside_bound=() if bound is None else encode_ranges(ranges - bound),
            target_runs=tuple(reached_targets),
        )


def encode_range(range_) -> EncodedRange:
    pc, *rest = range_
    return (pc.encode(), *rest)


def encode_ranges(ranges: set) -> tuple[EncodedRange, ...]:
    return tuple(sorted(encode_range(range_) for range_ in ranges))


def all_methods(suite: jpamb.Suite, *, eff: jpamb.Effect) -> list[jvm.AbsMethodID]:
    methods = {case.methodid.encode(): case.methodid for case in suite.cases(eff=eff)}
    return [methods[name] for name in sorted(methods)]


# The options fuzzer and fuzzer-extension share, and the methods they choose.
# domains are the names of the interval analysis' domains and the first is the default.
def parse_methods(parser: argparse.ArgumentParser, domains: tuple[str, ...] = ()):
    parser.add_argument(
        "methods", nargs="*", help="like 'jpamb.cases.Simple.assertFalse:()V'"
    )
    parser.add_argument("--all", action="store_true", help="every method in cases.txt")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--split", choices=["three", "boundary"], default="three")
    if domains:
        parser.add_argument("--domain", choices=domains, default=domains[0])
    parser.add_argument("--out", help="the JSON lines file, instead of stdout")
    args = parser.parse_args()
    suite, eff = jpamb.setup()
    if args.all:
        methods = all_methods(suite, eff=eff)
    else:
        methods = [jvm.AbsMethodID.decode(name) for name in args.methods]
    return args, jpamb.Bytecode(suite, eff, {}), methods


# bound_of(bc, methodid, split, domain) gives the ranges the interval analysis can
# reach, in the format of Trace.ranges. The analysis passes it in because it depends
# on the fuzzer.
def main(bound_of=None, domains: tuple[str, ...] = ()):
    parser = argparse.ArgumentParser(
        description="Fuzz JPAMB methods, one JSON row per method, version and seed."
    )
    parser.add_argument("--versions", default=",".join(Version))
    args, bc, methods = parse_methods(parser, domains)
    try:
        versions = [Version(name) for name in args.versions.split(",")]
    except ValueError:
        parser.error(f"--versions takes {','.join(Version)}, got {args.versions}")
    domain = args.domain if domains else ""

    # Absolute menace of a loop composition as this would never ever be acceptable
    # theoretically but it works practically.
    with open(args.out, "w") if args.out else contextlib.nullcontext(sys.stdout) as out:
        for methodid in methods:
            bound = None
            analysis_seconds = 0.0
            analysis_error = None
            if bound_of is not None:
                start = time.process_time()
                try:
                    bound = frozenset(bound_of(bc, methodid, args.split, domain))
                except Exception as error:
                    analysis_error = error
                analysis_seconds = round(time.process_time() - start, 3)
            for version in versions:
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
                            result = replace(
                                result, domain=domain, analysis_seconds=analysis_seconds
                            )
                        except Exception as crash:
                            # The interpreter and fuzz_input raise on what they do not
                            # support.
                            error = crash
                    if error is not None:
                        result = Result(
                            method=methodid.encode(),
                            version=version,
                            seed=seed,
                            split=args.split,
                            domain=domain,
                            stop_reason=Stop.CRASH,
                            error=repr(error),
                        )
                    out.write(json.dumps(asdict(result)) + "\n")
                    out.flush()
                    print(
                        f"{result.method} {version} {seed}: {result.stop_reason}",
                        file=sys.stderr,
                    )

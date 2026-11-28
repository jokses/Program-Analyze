import argparse
import contextlib
import json
import sys
from dataclasses import dataclass

import checks
import dynamic
import fuzzer
from unbounded import DOMAINS, Interval, RangeCoverageOracle, UnboundedIntervalAnalysis

import jpamb
import jvm

# The binary search of "Fuzz Smarter, Not Harder" stops at about 20 values and tests
# all of them. Going backwards tests at most as many for each kept input and int
# parameter.
CANDIDATES = 20
# Fuzzing for longer gets ten times as many runs without new coverage.
LONGER = 10 * fuzzer.MAX_RUNS_SINCE_NEW
# Each way gives the runs it needed to reach a target, or None. Combined goes
# backwards first and falls back on the binary search and counts the runs of both.
WAYS = ("backwards", "binary_search", "combined", "longer")


@dataclass(frozen=True)
class Attempt:
    runs: int
    reached: bool

    def runs_if_reached(self) -> int | None:
        return self.runs if self.reached else None


# Small enough to test every value in it.
def is_small(iv: Interval) -> bool:
    return iv.max - iv.min < CANDIDATES


def candidates(iv: Interval) -> list[int]:
    if is_small(iv):
        return list(range(iv.min, iv.max + 1))
    step = (iv.max - iv.min) / (CANDIDATES - 1)
    return sorted({round(iv.min + i * step) for i in range(CANDIDATES)})


def with_value(input: jpamb.case.Input, index: int, value: int) -> jpamb.case.Input:
    values = list(input.values)
    values[index] = jpamb.case.Int(value)
    return jpamb.case.Input(tuple(values))


# The method whose uncovered ranges are searched for, with the split and domain of
# its bound.
@dataclass(frozen=True)
class Search:
    bc: jpamb.Bytecode
    methodid: jvm.AbsMethodID
    split: str
    domain: str
    oracle: RangeCoverageOracle
    int_params: tuple[int, ...]


# Runs the values one by one until one reaches the target.
def try_values(
    search: Search, base: jpamb.case.Input, index: int, values: list[int], target
) -> Attempt:
    for runs, value in enumerate(values, start=1):
        input = with_value(base, index, value)
        trace = dynamic.run(
            search.bc,
            search.methodid,
            input,
            max_steps=fuzzer.MAX_STEPS,
            split=search.split,
        )
        if target in trace.ranges:
            return Attempt(runs, True)
    return Attempt(len(values), False)


def backwards(search: Search, kept: list[jpamb.case.Input], target) -> Attempt:
    runs = 0
    for base in kept:
        path = dynamic.run(
            search.bc, search.methodid, base, max_steps=fuzzer.MAX_STEPS
        ).path
        for index in search.int_params:
            iv = search.oracle.analysis.solve_uncovered_range(target, path, index)
            if iv.is_bot() or iv == Interval.top():
                continue
            tried = try_values(search, base, index, candidates(iv), target)
            runs += tried.runs
            if tried.reached:
                return Attempt(runs, True)
    return Attempt(runs, False)


# The halving does not depend on the seed, so the seeds share it.
HALVED: dict[tuple, Interval] = {}


# Source: Andersen et al., "Fuzz Smarter, Not Harder", PPPA'24 volume 1, section 2.
# Halve the interval of the parameter and keep a half where the analysis still
# allows the target. Their paper does not say which half to take when both allow it,
# and we take the lower one.
def halve(search: Search, index: int, target) -> Interval:
    key = (search.methodid, index, target, search.split, search.domain)
    if key not in HALVED:
        iv = Interval.top()
        while not is_small(iv):
            half = allowed_half(search, index, target, iv)
            if half is None:
                break
            iv = half
        HALVED[key] = iv
    return HALVED[key]


# The lower half of iv when the analysis allows the target there, else the upper
# half when it allows it there, else None.
def allowed_half(search: Search, index: int, target, iv: Interval) -> Interval | None:
    middle = (iv.min + iv.max) // 2
    for half in (Interval(iv.min, middle), Interval(middle + 1, iv.max)):
        analysis = UnboundedIntervalAnalysis(
            search.bc, search.methodid, param_ivs={index: half}, domain=search.domain
        )
        if target in analysis.run().reachable_range_tuples(split=search.split):
            return half
    return None


def binary_search(search: Search, base: jpamb.case.Input, target) -> Attempt:
    runs = 0
    for index in search.int_params:
        iv = halve(search, index, target)
        if is_small(iv):
            tried = try_values(search, base, index, candidates(iv), target)
            runs += tried.runs
            if tried.reached:
                return Attempt(runs, True)
    return Attempt(runs, False)


def extend(search: Search, seed: int) -> list[dict]:
    bound = frozenset(search.oracle.upper_bound)
    first = fuzzer.fuzz(
        search.bc,
        search.methodid,
        version=fuzzer.Version.RANGE,
        seed=seed,
        split=search.split,
        bound=bound,
    )
    covered = set(first.ranges)
    targets = [
        target
        for target in sorted(bound, key=str)
        if target[0].method == search.methodid
        and fuzzer.encode_range(target) not in covered
    ]
    # With the same seed the longer run repeats the first run and then goes on.
    longer = fuzzer.fuzz(
        search.bc,
        search.methodid,
        version=fuzzer.Version.RANGE,
        seed=seed,
        split=search.split,
        bound=bound,
        max_runs_since_new=LONGER,
        targets=frozenset(targets),
    )
    longer_runs = {target: run - first.runs for target, run in longer.target_runs}
    # The first input, from the empty code, is always kept, since every run covers an
    # instruction.
    kept = [jpamb.case.Input.decode(code) for code in first.kept_inputs]
    rows = []
    for target in targets:
        back = backwards(search, kept, target)
        binary = binary_search(search, kept[0], target)
        combined = (
            back if back.reached else Attempt(back.runs + binary.runs, binary.reached)
        )
        rows.append(
            {
                "method": search.methodid.encode(),
                "seed": seed,
                "split": search.split,
                "target": fuzzer.encode_range(target),
                "backwards": back.runs_if_reached(),
                "binary_search": binary.runs_if_reached(),
                "combined": combined.runs_if_reached(),
                "longer": longer_runs.get(fuzzer.encode_range(target)),
                "domain": search.domain,
            }
        )
    return rows


# The rows of targets, without the rows of methods that crashed.
def target_rows(rows) -> list[dict]:
    return [row for row in rows if "error" not in row]


def reached_runs(rows, way) -> list[int]:
    return [row[way] for row in rows if row[way] is not None]


def summary(rows, file):
    rows = target_rows(rows)
    print("targets", len(rows), file=file)
    for way in WAYS:
        runs = reached_runs(rows, way)
        print(way, "reached", len(runs), "median runs", checks.median(runs), file=file)


def main():
    parser = argparse.ArgumentParser(
        description="Try to reach the ranges the range fuzzer left uncovered."
    )
    args, bc, methods = fuzzer.parse_methods(parser, tuple(DOMAINS))

    rows = []
    with open(args.out, "w") if args.out else contextlib.nullcontext(sys.stdout) as out:
        for methodid in methods:
            params = methodid.extension.params
            int_params = [
                index
                for index, param in enumerate(params)
                if isinstance(param, jvm.Int)
            ]
            if not int_params:
                continue
            try:
                oracle = RangeCoverageOracle.build(
                    bc, methodid, args.split, args.domain
                )
                search = Search(
                    bc, methodid, args.split, args.domain, oracle, tuple(int_params)
                )
                new_rows = []
                for seed in range(args.seeds):
                    new_rows += extend(search, seed)
            # A crash in one method becomes an error row and the next method runs.
            except Exception as error:
                new_rows = [
                    {
                        "method": methodid.encode(),
                        "split": args.split,
                        "error": repr(error),
                        "domain": args.domain,
                    }
                ]
            for row in new_rows:
                out.write(json.dumps(row) + "\n")
            rows += new_rows
    summary(rows, sys.stderr)

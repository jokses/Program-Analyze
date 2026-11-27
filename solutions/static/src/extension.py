import argparse
import contextlib
import json
import statistics
import sys
from dataclasses import dataclass

import dynamic
import fuzzer
from unbounded import DOMAINS, Interval, RangeCoverageOracle, UnboundedIntervalAnalysis

import jpamb
import jvm

# Group 5 stop the binary search at about 20 values and test all of them. Going
# backwards tests at most as many for each kept input and int parameter.
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


def candidates(iv: Interval) -> list[int]:
    if iv.max - iv.min < CANDIDATES:
        return list(range(iv.min, iv.max + 1))
    step = (iv.max - iv.min) / (CANDIDATES - 1)
    return sorted({round(iv.min + i * step) for i in range(CANDIDATES)})


def with_value(input: jpamb.case.Input, index: int, value: int) -> jpamb.case.Input:
    values = list(input.values)
    values[index] = jpamb.case.Int(value)
    return jpamb.case.Input(tuple(values))


# Runs the values one by one until one reaches the target.
def try_values(bc, methodid, base, index, values, target, split) -> Attempt:
    for runs, value in enumerate(values, start=1):
        input = with_value(base, index, value)
        trace = dynamic.run(
            bc, methodid, input, max_steps=fuzzer.MAX_STEPS, split=split
        )
        if target in trace.ranges:
            return Attempt(runs, True)
    return Attempt(len(values), False)


def backwards(bc, methodid, oracle, kept, int_params, target, split) -> Attempt:
    runs = 0
    for base in kept:
        path = dynamic.run(bc, methodid, base, max_steps=fuzzer.MAX_STEPS).path
        for index in int_params:
            iv = oracle.analysis.solve_uncovered_range(target, path, index)
            if iv.is_bot() or iv == Interval.top():
                continue
            tried = try_values(bc, methodid, base, index, candidates(iv), target, split)
            runs += tried.runs
            if tried.reached:
                return Attempt(runs, True)
    return Attempt(runs, False)


# The halving does not depend on the seed, so the seeds share it.
HALVED: dict = {}


# Source: Andersen et al., "Fuzz Smarter, Not Harder", PPPA'24 volume 1, section 2.
# Halve the interval of the parameter and keep a half where the analysis still
# allows the target. Their paper does not say which half to take when both allow it,
# and we take the lower one.
def halve(bc, methodid, index, target, split) -> Interval:
    key = (methodid, index, target, split)
    if key not in HALVED:
        iv = Interval.top()
        while iv.max - iv.min >= CANDIDATES:
            middle = (iv.min + iv.max) // 2
            for half in (Interval(iv.min, middle), Interval(middle + 1, iv.max)):
                analysis = UnboundedIntervalAnalysis(bc, methodid, None, {index: half})
                if target in analysis.run().reachable_range_tuples(split=split):
                    iv = half
                    break
            else:
                break
        HALVED[key] = iv
    return HALVED[key]


def binary_search(bc, methodid, base, int_params, target, split) -> Attempt:
    runs = 0
    for index in int_params:
        iv = halve(bc, methodid, index, target, split)
        if iv.max - iv.min < CANDIDATES:
            tried = try_values(bc, methodid, base, index, candidates(iv), target, split)
            runs += tried.runs
            if tried.reached:
                return Attempt(runs, True)
    return Attempt(runs, False)


def encoded(target):
    pc, position, low, high = target
    return (pc.encode(), position, low, high)


def extend(bc, methodid, oracle, seed, split, int_params):
    bound = frozenset(oracle.upper_bound)
    first = fuzzer.fuzz(
        bc, methodid, version="range", seed=seed, split=split, bound=bound
    )
    covered = set(first.ranges)
    targets = [
        target
        for target in sorted(bound, key=str)
        if target[0].method == methodid and encoded(target) not in covered
    ]
    # With the same seed the longer run repeats the first run and then goes on.
    longer = fuzzer.fuzz(
        bc,
        methodid,
        version="range",
        seed=seed,
        split=split,
        bound=bound,
        max_runs_since_new=LONGER,
        goals=frozenset(targets),
    )
    longer_runs = {tuple(goal[:4]): goal[4] - first.runs for goal in longer.goal_runs}
    kept = [jpamb.case.Input.decode(code) for code in first.kept_inputs]
    rows = []
    for target in targets:
        back = backwards(bc, methodid, oracle, kept, int_params, target, split)
        binary = (
            binary_search(bc, methodid, kept[0], int_params, target, split)
            if kept
            else Attempt(0, False)
        )
        combined = (
            back if back.reached else Attempt(back.runs + binary.runs, binary.reached)
        )
        rows.append(
            {
                "method": methodid.encode(),
                "seed": seed,
                "split": split,
                "target": encoded(target),
                "backwards": back.runs_if_reached(),
                "binary_search": binary.runs_if_reached(),
                "combined": combined.runs_if_reached(),
                "longer": longer_runs.get(encoded(target)),
            }
        )
    return rows


def summary(rows, file):
    rows = [row for row in rows if "error" not in row]
    print("targets", len(rows), file=file)
    for way in WAYS:
        runs = [row[way] for row in rows if row[way] is not None]
        median = statistics.median(runs) if runs else "-"
        print(way, "reached", len(runs), "median runs", median, file=file)


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
                i for i, param in enumerate(params) if isinstance(param, jvm.Int)
            ]
            if not int_params:
                continue
            try:
                oracle = RangeCoverageOracle.build(
                    bc, methodid, args.split, args.domain
                )
                new_rows = []
                for seed in range(args.seeds):
                    new_rows += extend(
                        bc, methodid, oracle, seed, args.split, int_params
                    )
            # A crash in one method becomes an error row and the next method runs.
            except Exception as error:
                new_rows = [
                    {
                        "method": methodid.encode(),
                        "split": args.split,
                        "error": repr(error),
                    }
                ]
            for row in new_rows:
                out.write(json.dumps({**row, "domain": args.domain}) + "\n")
            rows += new_rows
    summary(rows, sys.stderr)

import argparse
import json
import math
import sys
from collections import defaultdict

import checks
import extension
import fuzzer
from unbounded import DOMAINS, UnboundedIntervalAnalysis

import jpamb
import jvm


# A method has unreachable code when the analysis never reaches one of its
# instructions or one of the case methods it calls. The analysis over-approximates
# and such an instruction cannot run.
def has_unreachable_code(bc, methodid, domain) -> bool:
    states = UnboundedIntervalAnalysis(bc, methodid, domain=domain).run().states
    for method in fuzzer.methods_reached(bc, methodid):
        for offset in range(len(bc.getmethod(method).opcodes)):
            if jvm.state.PC(method, offset) not in states:
                return True
    return False


# Source: https://www.stat.umn.edu/geyer/5601/examp/sign.html
# The p-value of the one-sided sign test: the probability that range wins at least
# this many of the paired seeds when both versions are equally good. A tie counts
# against range.
def sign_test(wins: int, pairs: int) -> float:
    if not pairs:
        return 1.0
    return sum(math.comb(pairs, k) for k in range(wins, pairs + 1)) / 2**pairs


# A pair is a run of instruction and a run of range with the same seed number. The
# random inputs still differ, because fuzz seeds them with the version too so the
# two runs of a pair are independent.
def paired(groups, method) -> list[tuple[dict, dict]]:
    instruction_rows = by_seed(groups[(method, fuzzer.Version.INSTRUCTION)])
    range_rows = by_seed(groups[(method, fuzzer.Version.RANGE)])
    seeds = sorted(instruction_rows.keys() & range_rows.keys())
    return [(instruction_rows[seed], range_rows[seed]) for seed in seeds]


def by_seed(rows) -> dict[int, dict]:
    return {row["seed"]: row for row in rows}


def rows_of(groups, version) -> list[dict]:
    rows = []
    for (_, row_version), runs in groups.items():
        if row_version == version:
            rows += runs
    return rows


def ranges_of(rows) -> set[tuple]:
    return {tuple(covered) for row in rows for covered in row["ranges"]}


def misses_an_outcome(row, expected) -> bool:
    return bool(expected[row["method"]] - set(row["outcomes"]))


# The rows of each method and version, without the crashes.
def group_runs(rows) -> dict[tuple[str, str], list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        if row["stop_reason"] != fuzzer.Stop.CRASH:
            groups[(row["method"], row["version"])].append(row)
    return groups


def unreachable_numbers(bc, groups, methods, domain):
    # Every version stops after one run on a method without parameters.
    unreachable = []
    for method in methods:
        methodid = jvm.AbsMethodID.decode(method)
        if methodid.extension.params and has_unreachable_code(bc, methodid, domain):
            unreachable.append(method)
    ratios = []
    earlier = pair_count = 0
    for method in unreachable:
        seed_pairs = paired(groups, method)
        # A version with only crash rows has no pair.
        if not seed_pairs:
            continue
        instruction_runs = [first["runs"] for first, _ in seed_pairs]
        range_runs = [second["runs"] for _, second in seed_pairs]
        instruction_median = checks.median(instruction_runs)
        range_median = checks.median(range_runs)
        ratios.append(instruction_median / range_median)
        for by_instruction, by_range in zip(instruction_runs, range_runs):
            earlier += by_range < by_instruction
        pair_count += len(seed_pairs)
        print(
            f"  {method}: instruction {instruction_median} runs"
            f" [{min(instruction_runs)}, {max(instruction_runs)}],"
            f" range {range_median} runs [{min(range_runs)}, {max(range_runs)}]"
        )
    print(
        "methods with parameters and unreachable code",
        len(unreachable),
        "median runs of instruction / range",
        checks.median(ratios),
    )
    print(
        "range stops earlier than instruction in",
        earlier,
        "of",
        pair_count,
        "paired seeds, sign test p =",
        f"{sign_test(earlier, pair_count):.2g}",
    )


def reaches_full_coverage(groups, method) -> bool:
    instruction_rows = groups[(method, fuzzer.Version.INSTRUCTION)]
    stops = {row["stop_reason"] for row in instruction_rows}
    return fuzzer.Stop.FULL_INSTRUCTION_COVERAGE in stops


def full_coverage_numbers(groups, methods):
    full = [method for method in methods if reaches_full_coverage(groups, method)]
    only_range = 0
    for method in full:
        random_rows = groups[(method, fuzzer.Version.RANDOM)]
        instruction_rows = groups[(method, fuzzer.Version.INSTRUCTION)]
        others = ranges_of(random_rows + instruction_rows)
        ranges = ranges_of(groups[(method, fuzzer.Version.RANGE)])
        only_range += bool(ranges - others)
    print(
        "methods where instruction reaches full coverage",
        len(full),
        "of them where range reaches ranges the others never reach",
        only_range,
    )
    # In a pair, range wins when it reaches more ranges that instruction
    # misses than the other way around.
    more = pair_count = 0
    for method in full:
        for instruction_row, range_row in paired(groups, method):
            by_instruction = ranges_of([instruction_row])
            by_range = ranges_of([range_row])
            more += len(by_range - by_instruction) > len(by_instruction - by_range)
            pair_count += 1
    print(
        "on those methods range reaches more ranges than instruction in",
        more,
        "of",
        pair_count,
        "paired seeds, sign test p =",
        f"{sign_test(more, pair_count):.2g}",
    )


def version_numbers(groups, expected):
    designed_by_method = defaultdict(list)
    for designed in json.loads(checks.DESIGNED_RANGES.read_text()):
        designed_by_method[designed["method"]].append(designed)
    for version in fuzzer.Version:
        version_rows = rows_of(groups, version)
        stops = defaultdict(int)
        for row in version_rows:
            stops[row["stop_reason"]] += 1
        missed = sum(misses_an_outcome(row, expected) for row in version_rows)
        designed_rows = []
        for row in version_rows:
            for designed in designed_by_method[row["method"]]:
                if checks.shown(designed, row["split"]):
                    designed_rows.append((row, designed))
        designed_missed = sum(
            not checks.reached(row, designed) for row, designed in designed_rows
        )
        print(
            version,
            "stops",
            dict(stops),
            "| runs after the last new coverage",
            checks.median(row["runs"] - row["last_new_run"] for row in version_rows),
            "| seeds missing a listed outcome",
            missed,
            "of",
            len(version_rows),
            "| designed ranges missed",
            designed_missed,
            "of",
            len(designed_rows),
        )


def bound_numbers(groups, expected):
    range_rows = rows_of(groups, fuzzer.Version.RANGE)
    # Stopping on the bound is only safe if no seed that did so missed an outcome.
    on_bound = [row for row in range_rows if row["stop_reason"] == fuzzer.Stop.BOUND]
    print(
        "range seeds that stopped on the bound and missed a listed outcome",
        sum(misses_an_outcome(row, expected) for row in on_bound),
        "of",
        len(on_bound),
    )
    covered = checks.median(
        (row["bound_size"] - row["bound_not_covered"]) / row["bound_size"]
        for row in range_rows
        if row["bound_size"]
    )
    print("median share of the bound the range version covers", covered)
    # Every run of the range version needs the bound and counts the analysis.
    analysis = sum(row["analysis_seconds"] for row in range_rows)
    total = sum(row["analysis_seconds"] + row["cpu_seconds"] for row in range_rows)
    print("analysis share of the range CPU time", f"{100 * analysis / total:.1f}%")
    outside = sum(len(row["outside_bound"]) for runs in groups.values() for row in runs)
    print("covered ranges outside the bound, all versions", outside)


def fuzzer_numbers(bc, rows, domain):
    groups = group_runs(rows)
    methods = sorted({method for method, _ in groups})
    print("methods", len(methods), "rows", sum(len(runs) for runs in groups.values()))
    unreachable_numbers(bc, groups, methods, domain)
    full_coverage_numbers(groups, methods)
    expected = checks.expected_outcomes()
    version_numbers(groups, expected)
    bound_numbers(groups, expected)


# For each designed range left uncovered, like n == 32 in divisionLoop, in how many
# seeds each way reached it and with how many runs.
def designed_numbers(rows):
    for designed in json.loads(checks.DESIGNED_RANGES.read_text()):
        targets = [
            row
            for row in extension.target_rows(rows)
            if checks.inside(row["target"], designed)
        ]
        # A split that cannot show the range gives no targets.
        if not targets:
            continue
        for way in extension.WAYS:
            runs = extension.reached_runs(targets, way)
            print(
                designed["what"],
                way,
                "reached in",
                len(runs),
                "of",
                len(targets),
                "seeds, median runs",
                checks.median(runs),
            )


def main():
    parser = argparse.ArgumentParser(
        description="Print the numbers of the evaluation for one split and domain."
    )
    parser.add_argument("runs", help="the JSON lines file of fuzzer-with-bound")
    parser.add_argument(
        "extension", nargs="?", help="the JSON lines file of fuzzer-extension"
    )
    args = parser.parse_args()
    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(suite, eff, {})
    runs = checks.read_rows(args.runs)
    _, domain = checks.split_and_domain(runs)
    if domain not in DOMAINS:
        parser.error(f"{args.runs} has no domain of fuzzer-with-bound: {domain!r}")
    fuzzer_numbers(bc, runs, domain)
    if args.extension:
        rows = checks.read_rows(args.extension)
        # The extension rows must have the split and domain of the runs.
        checks.split_and_domain(runs + rows)
        extension.summary(rows, sys.stdout)
        designed_numbers(rows)

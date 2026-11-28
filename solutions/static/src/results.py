import argparse
import math
import statistics
import sys
from collections import defaultdict

import checks
import estimates
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


# The methods with parameters and unreachable code. Every version stops after one
# run on a method without parameters, so those are left out.
def unreachable_methods(bc, methods, domain) -> list[str]:
    unreachable = []
    for method in methods:
        methodid = jvm.AbsMethodID.decode(method)
        if methodid.extension.params and has_unreachable_code(bc, methodid, domain):
            unreachable.append(method)
    return unreachable


def unreachable_numbers(bc, groups, methods, domain) -> list[str]:
    unreachable = unreachable_methods(bc, methods, domain)
    ratios = []
    earlier = pair_count = 0
    # The same test with one observation per method: its median runs.
    earlier_methods = 0
    # The methods with runs of both versions, for the estimates per method.
    compared = []
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
        earlier_methods += range_median < instruction_median
        compared.append(method)
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
    print(
        "range stops earlier than instruction in median runs on",
        earlier_methods,
        "of",
        len(ratios),
        "methods, sign test p =",
        f"{sign_test(earlier_methods, len(ratios)):.2g}",
    )
    return compared


def reaches_full_coverage(groups, method) -> bool:
    instruction_rows = groups[(method, fuzzer.Version.INSTRUCTION)]
    stops = {row["stop_reason"] for row in instruction_rows}
    return fuzzer.Stop.FULL_INSTRUCTION_COVERAGE in stops


def full_coverage_numbers(groups, methods) -> list[str]:
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
    return full


def version_numbers(groups, expected):
    for version in fuzzer.Version:
        version_rows = rows_of(groups, version)
        stops = defaultdict(int)
        for row in version_rows:
            stops[row["stop_reason"]] += 1
        missed = sum(misses_an_outcome(row, expected) for row in version_rows)
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
        )
        # A seed that hit the step limit may have found the outcome with more steps.
        missed_at_limit = sum(
            misses_an_outcome(row, expected) and "*" in row["outcomes"]
            for row in version_rows
        )
        print(
            f"  {version} seeds missing a listed outcome that hit the step limit",
            missed_at_limit,
        )
        seconds = checks.cpu_seconds(version_rows, version)
        print(
            f"  {version} CPU seconds per seed, median",
            round(checks.median(seconds), 3),
            "total",
            round(sum(seconds), 1),
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
    crashed = {row["method"] for row in rows if row["stop_reason"] == fuzzer.Stop.CRASH}
    print("methods left out because they crash", len(crashed - set(methods)))
    unreachable = unreachable_numbers(bc, groups, methods, domain)
    full = full_coverage_numbers(groups, methods)
    expected = checks.expected_outcomes()
    version_numbers(groups, expected)
    bound_numbers(groups, expected)
    method_numbers(groups, methods, unreachable, full, expected)


def interval_text(estimate: estimates.Estimate, scale) -> str:
    return (
        f"{scale(estimate.mean):.3g}, 95% CI [{scale(estimate.low):.3g},"
        f" {scale(estimate.high):.3g}], t-test over {estimate.methods} methods"
        f" p = {estimate.p_value:.2g}, Shapiro-Wilk p = {estimate.normality_p:.2g}"
    )


def ratio(log10_value: float) -> float:
    return 10**log10_value


def unchanged(value: float) -> float:
    return value


# The methods where the first version's values are below the second's, and above,
# by Mann-Whitney with Holm's correction over the methods of the claim.
def holm_counts(comparisons) -> tuple[int, int]:
    significant = [comparison for comparison in comparisons if comparison.holm_p < 0.05]
    below = sum(comparison.chance_below > 0.5 for comparison in significant)
    return below, len(significant) - below


# The same claims with the method as the unit: an estimate with its interval over
# the methods, and a test per method that needs no distribution.
def method_numbers(groups, methods, unreachable, full, expected):
    runs, runs_per_method = estimates.fewer_runs(groups, unreachable)
    print(
        "per method, unreachable code: instruction needs",
        interval_text(runs, ratio),
        "times the runs of range",
    )
    for comparison in runs_per_method:
        print(
            f"  {comparison.method}: chance range stops earlier"
            f" {comparison.chance_below:.2f}, Mann-Whitney p ="
            f" {comparison.p_value:.2g}, Holm p = {comparison.holm_p:.2g}"
        )
    earlier, later = holm_counts(runs_per_method)
    print(
        f"per method, unreachable code: range stops earlier with Holm p < 0.05 on"
        f" {earlier} of {len(runs_per_method)} methods, later on {later}"
    )

    ranges, ranges_per_method = estimates.more_ranges(groups, full)
    print(
        "per method, full instruction coverage: range covers",
        interval_text(ranges, unchanged),
        "more ranges per run",
    )
    differences = [
        statistics.fmean(comparison.second) - statistics.fmean(comparison.first)
        for comparison in ranges_per_method
    ]
    print(
        f"per method, full instruction coverage: range covers more on"
        f" {sum(difference > 0 for difference in differences)}, as many on"
        f" {sum(difference == 0 for difference in differences)}, fewer on"
        f" {sum(difference < 0 for difference in differences)} of"
        f" {len(differences)} methods"
    )
    more, fewer = holm_counts(ranges_per_method)
    print(
        f"per method, full instruction coverage: range covers more with Holm"
        f" p < 0.05 on {more} of {len(ranges_per_method)} methods, fewer on {fewer}"
    )

    cost = estimates.more_runs(groups, methods)
    cost_text = interval_text(cost, ratio)
    print("per method, range needs", cost_text, "times the runs of instruction")

    outside = {
        method
        for (method, _), rows in groups.items()
        if any(row["outside_bound"] for row in rows)
    }
    print(
        f"methods with a covered range outside the bound {len(outside)} of"
        f" {len(methods)}, 95% upper bound if none"
        f" {100 * estimates.zero_upper_bound(len(methods)):.2f}%"
    )
    range_rows = rows_of(groups, fuzzer.Version.RANGE)
    on_bound = [row for row in range_rows if row["stop_reason"] == fuzzer.Stop.BOUND]
    stopped = {row["method"] for row in on_bound}
    missed = {row["method"] for row in on_bound if misses_an_outcome(row, expected)}
    print(
        f"methods with a range seed that stopped on the bound and missed a listed"
        f" outcome {len(missed)} of {len(stopped)}, 95% upper bound if none"
        f" {100 * estimates.zero_upper_bound(len(stopped)):.2f}%"
    )


# Our proposal expects going backwards to reach the ranges the range version leaves
# uncovered on divisionLoop, n == 32 at its assertion.
DIVISION_LOOP = "jpamb.cases.Dependent.divisionLoop:(I)V"


def division_loop_targets(rows) -> list[dict]:
    targets = extension.target_rows(rows)
    return [row for row in targets if row["method"] == DIVISION_LOOP]


# For each way, how many of divisionLoop's uncovered ranges it reached, over all seeds,
# and with how many runs.
def division_loop_numbers(rows):
    targets = division_loop_targets(rows)
    for way in extension.WAYS:
        runs = extension.reached_runs(targets, way)
        print(
            "divisionLoop uncovered ranges",
            way,
            "reached",
            len(runs),
            "of",
            len(targets),
            "median runs",
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
        division_loop_numbers(rows)

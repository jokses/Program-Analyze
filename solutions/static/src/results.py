import argparse
import json
import math
import sys
from collections import defaultdict

import checks
import extension
import fuzzer
from unbounded import UnboundedIntervalAnalysis

import jpamb
import jvm


# fuzzer-results pairs the seeds and fuzzer-jvm-check builds the bound of one split
# and one domain, so a file that mixes them would give wrong numbers.
def split_and_domain(rows) -> tuple[str, str]:
    kinds = {(row["split"], row["domain"]) for row in rows}
    if len(kinds) != 1:
        raise ValueError(f"Expected one split and domain, got {sorted(kinds)}")
    return kinds.pop()


# A method has unreachable code when the analysis never reaches one of its
# instructions or one of the case methods it calls. The analysis over-approximates
# and such an instruction cannot run.
def has_unreachable_code(bc, methodid) -> bool:
    states = UnboundedIntervalAnalysis(bc, methodid).run().states
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


# A pair is one run of each version with the same seed number. The random inputs
# still differ, because fuzz seeds them with the version too so the two runs of a
# pair are independent.
def paired(groups, method, first, second):
    firsts = {row["seed"]: row for row in groups[(method, first)]}
    seconds = {row["seed"]: row for row in groups[(method, second)]}
    return [
        (firsts[seed], seconds[seed]) for seed in sorted(firsts.keys() & seconds.keys())
    ]


def rows_of(groups, version):
    return [
        row
        for (_, row_version), runs in groups.items()
        if row_version == version
        for row in runs
    ]


def fuzzer_numbers(bc, rows):
    groups = defaultdict(list)
    for row in rows:
        if row["stop_reason"] != "crash":
            groups[(row["method"], row["version"])].append(row)
    methods = sorted({method for method, _ in groups})
    print("methods", len(methods), "rows", sum(len(runs) for runs in groups.values()))

    # Every version stops after one run on a method without parameters.
    unreachable = [
        method
        for method in methods
        if jvm.AbsMethodID.decode(method).extension.params
        and has_unreachable_code(bc, jvm.AbsMethodID.decode(method))
    ]
    ratios = []
    earlier = pairs = 0
    for method in unreachable:
        seeds = paired(groups, method, "instruction", "range")
        # A version with only crash rows has no pair.
        if not seeds:
            continue
        instruction = [first["runs"] for first, _ in seeds]
        range_ = [second["runs"] for _, second in seeds]
        ratios.append(checks.median(instruction) / checks.median(range_))
        earlier += sum(
            range_runs < instruction_runs
            for instruction_runs, range_runs in zip(instruction, range_)
        )
        pairs += len(seeds)
        print(
            f"  {method}: instruction {checks.median(instruction)} runs"
            f" [{min(instruction)}, {max(instruction)}],"
            f" range {checks.median(range_)} runs [{min(range_)}, {max(range_)}]"
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
        pairs,
        "paired seeds, sign test p =",
        f"{sign_test(earlier, pairs):.2g}",
    )

    full = [
        method
        for method in methods
        if any(
            row["stop_reason"] == "full instruction coverage"
            for row in groups[(method, "instruction")]
        )
    ]
    only_range = 0
    for method in full:
        others = {
            tuple(covered)
            for version in ("random", "instruction")
            for row in groups[(method, version)]
            for covered in row["ranges"]
        }
        ranges = {
            tuple(covered)
            for row in groups[(method, "range")]
            for covered in row["ranges"]
        }
        only_range += bool(ranges - others)
    print(
        "methods where instruction reaches full coverage",
        len(full),
        "of them where range reaches ranges the others never reach",
        only_range,
    )
    # In a pair, range wins when it reaches more ranges that instruction
    # misses than the other way around.
    more = pairs = 0
    for method in full:
        for instruction, range_row in paired(groups, method, "instruction", "range"):
            by_instruction = {tuple(covered) for covered in instruction["ranges"]}
            by_range = {tuple(covered) for covered in range_row["ranges"]}
            more += len(by_range - by_instruction) > len(by_instruction - by_range)
            pairs += 1
    print(
        "on those methods range reaches more ranges than instruction in",
        more,
        "of",
        pairs,
        "paired seeds, sign test p =",
        f"{sign_test(more, pairs):.2g}",
    )

    expected = checks.expected_outcomes()
    designed_ranges = json.loads(checks.DESIGNED_RANGES.read_text())
    for version in fuzzer.VERSIONS:
        version_rows = rows_of(groups, version)
        stops = defaultdict(int)
        for row in version_rows:
            stops[row["stop_reason"]] += 1
        missed = sum(
            bool(expected[row["method"]] - set(row["outcomes"])) for row in version_rows
        )
        designed_rows = [
            (row, designed)
            for row in version_rows
            for designed in designed_ranges
            if designed["method"] == row["method"]
            and checks.shown(designed, row["split"])
        ]
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

    range_rows = rows_of(groups, "range")
    # Stopping on the bound is only safe if no seed that did so missed an outcome.
    on_bound = [
        row for row in range_rows if row["stop_reason"] == "ranges meet the bound"
    ]
    print(
        "range seeds that stopped on the bound and missed a listed outcome",
        sum(bool(expected[row["method"]] - set(row["outcomes"])) for row in on_bound),
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


# For each designed range left uncovered, like n == 32 in divisionLoop, in how many
# seeds each way reached it and with how many runs.
def designed_numbers(rows):
    for designed in json.loads(checks.DESIGNED_RANGES.read_text()):
        targets = [
            row
            for row in rows
            if "error" not in row and checks.inside(row["target"], designed)
        ]
        # A split that cannot show the range gives no targets.
        if not targets:
            continue
        for way in extension.WAYS:
            runs = [row[way] for row in targets if row[way] is not None]
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
    split_and_domain(runs)
    fuzzer_numbers(bc, runs)
    if args.extension:
        rows = checks.read_rows(args.extension)
        split_and_domain(runs + rows)
        extension.summary(rows, sys.stdout)
        designed_numbers(rows)

import json
import statistics
import sys
from collections import defaultdict

import checks
import extension
import fuzzer
from unbounded import UnboundedIntervalAnalysis

import jpamb
import jvm


def load(path):
    with open(path) as lines:
        return [json.loads(line) for line in lines]


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


def median(values):
    values = list(values)
    return statistics.median(values) if values else "-"


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
    for method in unreachable:
        instruction = median(row["runs"] for row in groups[(method, "instruction")])
        range_ = median(row["runs"] for row in groups[(method, "range")])
        # A version with only crash rows has no median.
        if "-" in (instruction, range_):
            continue
        ratios.append(instruction / range_)
        print(f"  {method}: instruction {instruction} runs, range {range_} runs")
    print(
        "methods with parameters and unreachable code",
        len(unreachable),
        "median runs of instruction / range",
        median(ratios),
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
            median(row["runs"] - row["last_new_run"] for row in version_rows),
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
    covered = median(
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


# fuzzer-results RUNS.jsonl [EXTENSION.jsonl] prints the numbers the proposal asks for.
def main():
    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(suite, eff, {})
    fuzzer_numbers(bc, load(sys.argv[1]))
    if len(sys.argv) > 2:
        extension.summary(load(sys.argv[2]), sys.stdout)

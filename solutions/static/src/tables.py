import json
import re
import statistics
from pathlib import Path

import checks
import estimates
import extension
import fuzzer
import results

# The paper's tables, as LaTeX for \input. Every cell comes from the same estimates
# as the summaries and the figures, so no number is typed by hand.

SPLITS = ("three", "boundary")
VERSIONS = (fuzzer.Version.RANGE, fuzzer.Version.INSTRUCTION, fuzzer.Version.RANDOM)
ROW_END = r" \\"

# The methods we added that have a failing input.
OWN_METHODS = (
    "Loops.decrementLoop",
    "Loops.multiplicationLoop",
    "Loops.nestedLoop",
    "Arrays.arrayIndexValue",
    "Arrays.arrayLoop",
    "Arrays.arraySumIsLargeInt",
    "MagicConstants.loopAndEquality",
    "MagicConstants.narrowEquality",
    "MagicConstants.rangeCondition",
)


def ratio_cell(estimate: estimates.Estimate) -> str:
    mean, low, high = 10**estimate.mean, 10**estimate.low, 10**estimate.high
    return rf"${mean:.3g}\times$ [{low:.3g}, {high:.3g}]"


def difference_cell(estimate: estimates.Estimate) -> str:
    return f"{estimate.mean:.1f} [{estimate.low:.1f}, {estimate.high:.1f}]"


# A zero count gets the highest rate still consistent with it.
def count_cell(count: int, total: int) -> str:
    if count:
        return f"{count:,} of {total:,}"
    bound = 100 * estimates.zero_upper_bound(total)
    return rf"0 of {total:,} ($\le$ {bound:.1f}\%)"


# The numbers of a fuzzer-jvm-check summary: same and different outcomes, divisors
# and divisors outside the bound.
def jvm_numbers(path: Path) -> tuple[int, int, int, int]:
    text = path.read_text()
    outcomes = re.search(r"same outcome (\d+) .* different (\d+)", text)
    divisors = re.search(r"divisors (\d+) \| outside the bound (\d+)", text)
    same, different = map(int, outcomes.groups())
    seen, outside = map(int, divisors.groups())
    return same, different, seen, outside


# n == 32 at the assertion of divisionLoop: in how many seeds going backwards
# reached it, of the seeds that left it uncovered.
def designed_backwards(targets) -> tuple[int, int]:
    division_loop = json.loads(checks.DESIGNED_RANGES.read_text())[0]
    open_ranges = [
        row for row in targets if checks.inside(row["target"], division_loop)
    ]
    reached = extension.reached_runs(open_ranges, "backwards")
    return len(reached), len(open_ranges)


def more_same_fewer(comparisons) -> str:
    differences = [
        statistics.fmean(comparison.second) - statistics.fmean(comparison.first)
        for comparison in comparisons
    ]
    more = sum(difference > 0 for difference in differences)
    same = sum(difference == 0 for difference in differences)
    return f"{more} / {same} / {len(differences) - more - same}"


# The rows of the claims table for one split, as (what, cell). split_run holds one
# split's groups, methods, the methods with unreachable code and with full instruction
# coverage, the extension's uncovered ranges and the JVM numbers.
def claims_rows(split_run: dict, expected) -> list[tuple[str, str]]:
    groups = split_run["groups"]
    methods = split_run["methods"]
    outside = {
        method
        for (method, _), rows in groups.items()
        if any(row["outside_bound"] for row in rows)
    }
    range_rows = results.rows_of(groups, fuzzer.Version.RANGE)
    on_bound = [row for row in range_rows if row["stop_reason"] == fuzzer.Stop.BOUND]
    missed = [row for row in on_bound if results.misses_an_outcome(row, expected)]
    stopped_methods = {row["method"] for row in on_bound}
    missed_methods = {row["method"] for row in missed}
    full, full_comparisons = estimates.more_ranges(groups, split_run["full"])
    runs, runs_comparisons = estimates.fewer_runs(groups, split_run["unreachable"])
    earlier, later = results.holm_counts(runs_comparisons)
    analysis = sum(row["analysis_seconds"] for row in range_rows)
    total = sum(row["analysis_seconds"] + row["cpu_seconds"] for row in range_rows)
    backwards, open_seeds = designed_backwards(split_run["extension"])
    same, different, divisors, divisors_outside = split_run["jvm"]
    cost = estimates.more_runs(groups, methods)
    return [
        (
            "Methods with a covered range outside the bound",
            count_cell(len(outside), len(methods)),
        ),
        (
            "Methods with a stop on the bound that missed an outcome",
            count_cell(len(missed_methods), len(stopped_methods)),
        ),
        (r"\quad the same, in seeds", count_cell(len(missed), len(on_bound))),
        (
            "Full coverage: more ranges per run, range $-$ instruction",
            difference_cell(full),
        ),
        (
            r"\quad methods with more / as many / fewer",
            more_same_fewer(full_comparisons),
        ),
        (
            r"\texttt{divisionLoop}: $n = 32$ reached by going backwards",
            f"{backwards} of {open_seeds} seeds",
        ),
        (
            "Share of the range version's CPU time in the analysis",
            rf"{100 * analysis / total:.1f}\%",
        ),
        ("Unreachable code: runs of instruction / runs of range", ratio_cell(runs)),
        (
            r"\quad methods where range is earlier / later (Holm)",
            f"{earlier} / {later} of {len(runs_comparisons)}",
        ),
        ("Cost: runs of range / runs of instruction, all methods", ratio_cell(cost)),
        (
            "JVM: kept inputs with another outcome",
            count_cell(different, same + different),
        ),
        ("JVM: divisors outside the bound", count_cell(divisors_outside, divisors)),
    ]


def claims_table(run: dict, expected, path: Path):
    rows_by_split = {split: claims_rows(run[split], expected) for split in SPLITS}
    # The first column wraps, so the table keeps to the width of the text.
    lines = [
        r"\begin{tabular}{@{}p{0.46\linewidth}ll@{}}",
        r"\toprule",
        r"Claim & \texttt{three} & \texttt{boundary}" + ROW_END,
        r"\midrule",
    ]
    three, boundary = rows_by_split["three"], rows_by_split["boundary"]
    for (what, cell), (_, other) in zip(three, boundary):
        lines.append(f"{what} & {cell} & {other}" + ROW_END)
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


# For each method we added with a failing input: in how many seeds each version finds
# the failing input, per split.
def own_methods_table(run: dict, path: Path):
    body = []
    seed_counts = set()
    for name in OWN_METHODS:
        cells = []
        for split in SPLITS:
            groups = run[split]["groups"]
            method = next(
                key for key, _ in groups if key.startswith(f"jpamb.cases.{name}:")
            )
            for version in VERSIONS:
                rows = groups[(method, version)]
                seed_counts.add(len(rows))
                found = sum("assertion error" in row["outcomes"] for row in rows)
                cells.append(str(found))
        method_name = name.split(".")[1]
        body.append(rf"\texttt{{{method_name}}} & " + " & ".join(cells) + ROW_END)
    # One seed count for every cell, so the header can say it once.
    [seeds] = seed_counts
    header = " & ".join(rf"\texttt{{{version}}}" for version in VERSIONS)
    columns = "@{}l" + "r" * len(SPLITS) * len(VERSIONS) + "@{}"
    width = len(VERSIONS)
    splits = " & ".join(
        rf"\multicolumn{{{width}}}{{c}}{{\texttt{{{split}}}, of {seeds} seeds}}"
        for split in SPLITS
    )
    lines = [
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        f" & {splits}" + ROW_END,
        f"Method & {header} & {header}" + ROW_END,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n")

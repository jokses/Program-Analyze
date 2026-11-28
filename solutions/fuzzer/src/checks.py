import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import fuzzer
from dynamic import comparisons_in_method, range_of, split_points

import jpamb
import jvm

# The ranges our methods are written to reach. The pcs are from `jpamb inspect`.
DESIGNED_RANGES = Path(__file__).resolve().parents[1] / "designed_ranges.json"


def read_rows(path: str) -> list[dict]:
    with open(path) as lines:
        return [json.loads(line) for line in lines]


# fuzzer-results pairs the seeds and fuzzer-jvm-check builds the bound of one split
# and one domain, so a file that mixes them would give wrong numbers.
def split_and_domain(rows) -> tuple[str, str]:
    kinds = {(row["split"], row["domain"]) for row in rows}
    if len(kinds) != 1:
        raise ValueError(f"Expected one split and domain, got {sorted(kinds)}")
    return kinds.pop()


# The median, or "-" when there are no values.
def median(values):
    values = list(values)
    return statistics.median(values) if values else "-"


def expected_outcomes() -> dict[str, set[str]]:
    suite, eff = jpamb.setup()
    expected = defaultdict(set)
    for case in suite.cases(eff=eff):
        expected[case.methodid.encode()].add(case.result)
    return expected


# A split shows a designed range when its range around the low end lies inside it.
# With "three" the range around 301 is [301, 2**31 - 1] and only "boundary" shows
# sum == 301.
def shown(designed, split) -> bool:
    methodid = jvm.AbsMethodID.decode(designed["method"])
    points = split_points(comparisons_in_method(methodid), split)
    low, high = range_of(designed["low"], points)
    return within(designed, low, high)


def within(designed, low: int, high: int) -> bool:
    return designed["low"] <= low and high <= designed["high"]


def inside(covered, designed) -> bool:
    pc, operand, low, high = covered
    designed_pc = f"{designed['method']}:{designed['pc']}"
    same_operand = pc == designed_pc and operand == designed["operand"]
    return same_operand and within(designed, low, high)


# Only range uses the bound, so only its seconds include the analysis.
def cpu_seconds(rows, version) -> list[float]:
    if version == fuzzer.Version.RANGE:
        return [row["cpu_seconds"] + row["analysis_seconds"] for row in rows]
    return [row["cpu_seconds"] for row in rows]


def reached(row, designed) -> bool:
    return any(inside(covered, designed) for covered in row["ranges"])


def main():
    parser = argparse.ArgumentParser(
        description="Summarize fuzzer rows per method, split, domain and version."
    )
    parser.add_argument(
        "rows", nargs="+", help="JSON lines files of fuzzer or fuzzer-with-bound"
    )
    args = parser.parse_args()
    rows = []
    for path in args.rows:
        rows += read_rows(path)
    expected = expected_outcomes()
    designed_ranges = defaultdict(list)
    for designed in json.loads(DESIGNED_RANGES.read_text()):
        designed_ranges[designed["method"]].append(designed)

    groups = defaultdict(list)
    crashes = defaultdict(int)
    for row in rows:
        if row["stop_reason"] == fuzzer.Stop.CRASH:
            crashes[(row["method"], row["error"])] += 1
        else:
            # fuzzer writes an empty domain and no bound, so its rows stay apart from
            # the rows of fuzzer-with-bound.
            domain = row["domain"] or "-"
            key = (row["method"], row["split"], domain, row["version"])
            groups[key].append(row)
    for (method, error), count in sorted(crashes.items()):
        print(fuzzer.Stop.CRASH, method, count, "rows", error)

    print("method split domain version seeds runs seconds stops missed bound outside")
    for (method, split, domain, version), seed_rows in sorted(groups.items()):
        stops = defaultdict(int)
        missed = defaultdict(int)
        for row in seed_rows:
            stops[row["stop_reason"]] += 1
            # A run that hit the step limit may have reached the outcome later.
            hit_step_limit = "*" in row["outcomes"]
            for outcome in expected[method] - set(row["outcomes"]):
                label = f"{outcome} (with *)" if hit_step_limit else outcome
                missed[label] += 1
        seconds = cpu_seconds(seed_rows, version)
        bound = "-"
        bound_size = seed_rows[0]["bound_size"]
        if bound_size is not None:
            covered = [
                row["bound_size"] - row["bound_not_covered"] for row in seed_rows
            ]
            bound = f"{median(covered)}/{bound_size}"
        outside = sum(len(row["outside_bound"]) for row in seed_rows)
        print(
            method,
            split,
            domain,
            version,
            len(seed_rows),
            median(row["runs"] for row in seed_rows),
            round(median(seconds), 3),
            dict(stops),
            dict(missed),
            bound,
            outside,
        )
        for designed in designed_ranges[method]:
            if shown(designed, split):
                count = sum(reached(row, designed) for row in seed_rows)
                print(f"  {designed['what']} in {count} of {len(seed_rows)} seeds")
            else:
                print(f"  {designed['what']} cannot be seen with {split}")

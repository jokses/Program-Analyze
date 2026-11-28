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


# The median, or - without values, like for a way that reached no target.
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
    return designed["low"] <= low and high <= designed["high"]


def inside(covered, designed) -> bool:
    site, position, low, high = covered
    return (
        site == f"{designed['method']}:{designed['pc']}"
        and position == designed["operand"]
        and designed["low"] <= low
        and high <= designed["high"]
    )


def reached(row, designed) -> bool:
    return any(inside(covered, designed) for covered in row["ranges"])


def main():
    parser = argparse.ArgumentParser(
        description="Summarize fuzzer rows per method, split, domain and version."
    )
    parser.add_argument(
        "rows", nargs="+", help="JSON lines files of fuzzer or fuzzer-with-bound"
    )
    rows = [row for path in parser.parse_args().rows for row in read_rows(path)]
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
    for (method, split, domain, version), runs in sorted(groups.items()):
        stops = defaultdict(int)
        missed = defaultdict(int)
        for row in runs:
            stops[row["stop_reason"]] += 1
            for outcome in expected[method] - set(row["outcomes"]):
                # A run that hit the step limit may have reached the outcome later.
                missed[outcome + (" (with *)" if "*" in row["outcomes"] else "")] += 1
        uses_bound = version == fuzzer.Version.RANGE
        seconds = [
            row["cpu_seconds"] + (row["analysis_seconds"] if uses_bound else 0)
            for row in runs
        ]
        bound = "-"
        if runs[0]["bound_size"] is not None:
            covered = [row["bound_size"] - row["bound_not_covered"] for row in runs]
            bound = f"{median(covered)}/{runs[0]['bound_size']}"
        outside = sum(len(row["outside_bound"]) for row in runs)
        print(
            method,
            split,
            domain,
            version,
            len(runs),
            median(row["runs"] for row in runs),
            round(median(seconds), 3),
            dict(stops),
            dict(missed),
            bound,
            outside,
        )
        for designed in designed_ranges[method]:
            if shown(designed, split):
                count = sum(reached(row, designed) for row in runs)
                print(f"  {designed['what']} in {count} of {len(runs)} seeds")
            else:
                print(f"  {designed['what']} cannot be seen with {split}")

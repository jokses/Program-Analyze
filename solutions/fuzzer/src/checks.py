import json
import statistics
import sys
from collections import defaultdict

import jpamb


def expected_outcomes() -> dict[str, set[str]]:
    suite, eff = jpamb.setup()
    expected = defaultdict(set)
    for case in suite.cases(eff=eff):
        expected[case.methodid.encode()].add(case.result)
    return expected


def main():
    rows = []
    for file in sys.argv[1:]:
        with open(file) as lines:
            rows += [json.loads(line) for line in lines]
    expected = expected_outcomes()

    groups = defaultdict(list)
    crashes = defaultdict(int)
    for row in rows:
        if row["stop_reason"] == "crash":
            crashes[(row["method"], row["error"])] += 1
        else:
            groups[(row["method"], row["split"], row["version"])].append(row)
    for (method, error), count in sorted(crashes.items()):
        print("crash", method, count, "rows", error)

    print("method split version seeds runs seconds stops missed bound outside")
    for (method, split, version), runs in sorted(groups.items()):
        stops = defaultdict(int)
        missed = defaultdict(int)
        for row in runs:
            stops[row["stop_reason"]] += 1
            for outcome in expected[method] - set(row["outcomes"]):
                # A run that hit the step limit may have reached the outcome later.
                missed[outcome + (" (with *)" if "*" in row["outcomes"] else "")] += 1
        # Only the range version uses the bound.
        seconds = [
            row["cpu_seconds"] + (row["analysis_seconds"] if version == "range" else 0)
            for row in runs
        ]
        bound = "-"
        if runs[0]["bound_size"] is not None:
            covered = [row["bound_size"] - row["bound_not_covered"] for row in runs]
            bound = f"{statistics.median(covered)}/{runs[0]['bound_size']}"
        outside = sum(len(row["outside_bound"]) for row in runs)
        print(
            method,
            split,
            version,
            len(runs),
            statistics.median(row["runs"] for row in runs),
            round(statistics.median(seconds), 3),
            dict(stops),
            dict(missed),
            bound,
            outside,
        )

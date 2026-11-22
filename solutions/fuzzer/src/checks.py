import json
import sys

from dynamic import to_int

import jpamb

DIVISION_LOOP = "jpamb.cases.Dependent.divisionLoop:(I)V"
ARRAY_SUM_IS_LARGE = "jpamb.cases.Arrays.arraySumIsLarge:([I)V"
# n != 32 and sum > 300, found in target/decompiled/jpamb/cases
DIVISION_LOOP_ASSERTION = DIVISION_LOOP + ":15"
ARRAY_SUM_COMPARISON = ARRAY_SUM_IS_LARGE + ":20"


def main():
    rows = []
    for file in sys.argv[1:]:
        with open(file) as lines:
            rows += [json.loads(line) for line in lines]

    for row in rows:
        name = (row["split"], row["version"], row["seed"])
        if row["stop_reason"] in ("crash", "max runs"):
            print(row["method"], *name, row["stop_reason"], row["error"])
        elif row["method"] == DIVISION_LOOP:
            covered = sum(pc.startswith(DIVISION_LOOP) for pc in row["instructions"])
            reached = [DIVISION_LOOP_ASSERTION, 0, 32, 32] in row["ranges"]
            print("divisionLoop", *name, covered, reached, row["stop_reason"])
        elif row["method"] == ARRAY_SUM_IS_LARGE:
            sums = [
                (low, high)
                for pc, position, low, high in row["ranges"]
                if pc == ARRAY_SUM_COMPARISON and position == 0
            ]
            # Java ints wrap around, also in the sum.
            kept = [
                to_int(sum(value.values))
                for kept_input in row["kept_inputs"]
                for value in jpamb.case.Input.decode(kept_input).values
            ]
            print("arraySumIsLarge", *name, sums, 301 in kept)

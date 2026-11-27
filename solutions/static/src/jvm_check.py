import subprocess
import sys
from collections import defaultdict

import dynamic
import fuzzer
import jvm_agent
from unbounded import RangeCoverageOracle

import jpamb
import jvm
from jvm.state import PC
from results import load

# A JVM run that takes longer than this is *, like a run over the interpreter's steps.
JVM_TIMEOUT = 5.0


# A divisor site like jpamb.cases.Simple.divideByN(I)I#div_rhs@6 is the idiv or irem
# at bytecode offset 6 which is our pc with that offset.
def divisor_pc(bc, site: str) -> PC | None:
    method, kind = site.split("#")
    if not kind.startswith("div_rhs@"):
        return None
    methodid = jvm.AbsMethodID.decode(method.replace("(", ":(", 1))
    offset = int(kind.removeprefix("div_rhs@"))
    opcodes = bc.getmethod(methodid).opcodes
    return PC(
        methodid, next(i for i, opr in enumerate(opcodes) if opr.offset == offset)
    )


# Like java.lang.RuntimeException: Unexpected from Exception in thread "main" ... on
# stderr. jpamb.Runtime fails like that on an exception it has no outcome for.
def java_exception(stderr: str) -> str:
    line = next(line for line in stderr.splitlines() if "Exception in thread" in line)
    return line.split('" ', 1)[1]


def divisor_ranges(bound, pc: PC) -> list[tuple[int, int]]:
    return sorted(
        (low, high) for at, position, low, high in bound if at == pc and position == 1
    )


# True when the ranges leave no value from low to high out.
def covers(ranges, low: int, high: int) -> bool:
    for start, end in ranges:
        if start > low:
            break
        low = max(low, end + 1)
    return low > high


# fuzzer-jvm-check RUNS.jsonl runs every kept input of fuzzer-with-bound again on the
# JVM with the agent, and compares the outcome and the divisors with ours.
def main():
    rows = [row for row in load(sys.argv[1]) if row["stop_reason"] != "crash"]
    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(suite, eff, {})
    kept = defaultdict(set)
    for row in rows:
        kept[row["method"]].update(row["kept_inputs"])
    split, domain = rows[0]["split"], rows[0]["domain"]

    same, our_limits, differ = 0, 0, []
    # Why the JVM gave no outcome for an input, and the inputs.
    refused = defaultdict(list)
    # The agent keeps the lowest and highest divisor of a site and how often it ran. After
    # three runs or more, a divisor between them may be in a hole of the bound.
    divisors, undecided, outside = 0, [], []
    for name, codes in sorted(kept.items()):
        methodid = jvm.AbsMethodID.decode(name)
        bound = RangeCoverageOracle.build(bc, methodid, split, domain).upper_bound
        for code in sorted(codes):
            # jpamb writes the char ' as ''' which it cannot read back.
            try:
                input = jpamb.case.Input.decode(code)
            except ValueError:
                refused["jpamb cannot read the input back"].append(code)
                continue
            ours = dynamic.run(bc, methodid, input, max_steps=fuzzer.MAX_STEPS).outcome
            try:
                run = jvm_agent.run_jvm_dynamic_with_ranges(
                    methodid, input, timeout_s=JVM_TIMEOUT
                )
            except subprocess.CalledProcessError as error:
                refused[java_exception(error.stderr)].append(code)
                continue
            # MAX_STEPS and MAX_ARRAY_SIZE stop our interpreter but not the JVM.
            if ours in ("*", "out of memory"):
                our_limits += 1
            elif ours == run.outcome:
                same += 1
            else:
                differ.append((name, code, ours, run.outcome))
            for site, (low, high, count) in run.int_ranges.items():
                pc = divisor_pc(bc, site)
                if pc is None:
                    continue
                divisors += 1
                ranges = divisor_ranges(bound, pc)
                if covers(ranges, low, low) and covers(ranges, high, high):
                    if count > 2 and not covers(ranges, low, high):
                        undecided.append((pc.encode(), low, high, code))
                else:
                    outside.append((pc.encode(), low, high, code))
    ran = same + our_limits + len(differ)
    print(f"inputs run on the JVM {ran} | same outcome {same}", end=" ")
    print(f"| over our limits {our_limits} | different {len(differ)}")
    for difference in differ:
        print("  different:", *difference)
    for reason, codes in refused.items():
        print(f"no outcome from the JVM, {reason}: {len(codes)}, like {codes[0]!r}")
    print(f"divisor sites {divisors} | outside the bound {len(outside)}", end=" ")
    print(f"| undecided {len(undecided)}")
    for site in outside:
        print("  outside:", *site)
    for site in undecided:
        print("  undecided:", *site)

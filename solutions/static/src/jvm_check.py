import argparse
import subprocess
from collections import defaultdict

import checks
import dynamic
import fuzzer
import jvm_agent
from results import split_and_domain
from unbounded import RangeCoverageOracle

import jpamb
import jvm
from jvm.state import PC

# A JVM run that takes longer than this is *, like a run over the interpreter's steps.
JVM_TIMEOUT = 5.0

DIVISIONS = (jvm.BinaryOpr.Div, jvm.BinaryOpr.Rem)


# The agent numbers the instructions of a method from 0 like jpamb does, so its index
# is our pc. Only an if with two operands and an idiv or irem have an operand 1.
def divisor_pc(bc, method: str, index: int) -> PC | None:
    methodid = jvm.AbsMethodID.decode(method)
    opcode = bc.getmethod(methodid).opcodes[index]
    if isinstance(opcode, jvm.Binary) and opcode.operant in DIVISIONS:
        return PC(methodid, index)
    assert isinstance(opcode, jvm.If), f"The agent's {method} {index} is {opcode}"
    return None


# Like java.lang.RuntimeException: Unexpected from Exception in thread "main" ... on
# stderr. jpamb.Runtime fails like that on an exception it has no outcome for.
def java_exception(stderr: str) -> str:
    line = next(line for line in stderr.splitlines() if "Exception in thread" in line)
    return line.split('" ', 1)[1]


def in_bound(bound, pc: PC, value: int) -> bool:
    return any(
        at == pc and position == 1 and low <= value <= high
        for at, position, low, high in bound
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run every kept input of fuzzer-with-bound again on the JVM with"
        " the agent, and compare the outcome and the divisors with ours."
    )
    parser.add_argument("runs", help="the JSON lines file of fuzzer-with-bound")
    rows = checks.read_rows(parser.parse_args().runs)
    rows = [row for row in rows if row["stop_reason"] != fuzzer.Stop.CRASH]
    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(suite, eff, {})
    kept = defaultdict(set)
    for row in rows:
        kept[row["method"]].update(row["kept_inputs"])
    split, domain = split_and_domain(rows)

    same, our_limits, differ = 0, 0, []
    # Why the JVM gave no outcome for an input, and the inputs.
    refused = defaultdict(list)
    divisors, outside = 0, []
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
                observation = jvm_agent.run_with_agent(
                    methodid, input, timeout_seconds=JVM_TIMEOUT
                )
            except subprocess.CalledProcessError as error:
                refused[java_exception(error.stderr)].append(code)
                continue
            # MAX_STEPS and MAX_ARRAY_SIZE stop our interpreter but not the JVM.
            if ours in ("*", "out of memory"):
                our_limits += 1
            elif ours == observation.outcome:
                same += 1
            else:
                differ.append((name, code, ours, observation.outcome))
            for (method, index, position), values in observation.values.items():
                pc = divisor_pc(bc, method, index) if position == 1 else None
                if pc is None:
                    continue
                for value in set(values):
                    divisors += 1
                    if not in_bound(bound, pc, value):
                        outside.append((pc.encode(), value, code))
    ran = same + our_limits + len(differ)
    print(f"inputs run on the JVM {ran} | same outcome {same}", end=" ")
    print(f"| over our limits {our_limits} | different {len(differ)}")
    for difference in differ:
        print("  different:", *difference)
    for reason, codes in refused.items():
        print(f"no outcome from the JVM, {reason}: {len(codes)}, like {codes[0]!r}")
    print(f"divisors {divisors} | outside the bound {len(outside)}")
    for divisor in outside:
        print("  outside:", *divisor)

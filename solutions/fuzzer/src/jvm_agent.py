import random
import subprocess
import time
from dataclasses import dataclass

import dynamic

import jpamb
import jvm


@dataclass
class AgentObservation:
    outcome: str
    sites: set[str]
    int_ranges: dict[str, tuple[int, int, int]]  # site -> (min, max, count)


# Built with the commands in solutions/fuzzer/README.md.
AGENT_JAR = "build/jvm/agent.jar"
CLASSES = "build/jvm/classes"


def parse_agent_int_ranges(payload: str) -> dict[str, tuple[int, int, int]]:
    ranges: dict[str, tuple[int, int, int]] = {}
    for entry in filter(None, payload.split("|")):
        site, spec = entry.rsplit("=", 1)
        bounds, count = spec.rsplit("#", 1)
        low, high = bounds.split("..")
        ranges[site] = (int(low), int(high), int(count))
    return ranges


# jpamb.Runtime stops at the first exception so each JVM runs one input. A run that
# does not end in time is *.
def run_jvm_dynamic_with_ranges(
    methodid: jvm.AbsMethodID, input: jpamb.case.Input, *, timeout_s: float
) -> AgentObservation:
    command = [
        "java",
        "-ea",
        "-XX:+TieredCompilation",
        "-XX:TieredStopAtLevel=1",
        # The JVM logs its warnings to stdout by default, and here stdout has only the
        # outcome and the agent's two lines.
        # Source: https://docs.oracle.com/en/java/javase/25/docs/specs/man/java.html#enable-logging-with-the-jvm-unified-logging-framework
        "-Xlog:disable",
        "-Xlog:all=warning:stderr",
        f"-javaagent:{AGENT_JAR}",
        "-cp",
        CLASSES,
        "jpamb.Runtime",
        methodid.encode(),
        # A command line argument cannot hold a NUL so jpamb.utils.InputParser reads
        # \0 as one and \\ as a backslash.
        input.encode().replace("\\", "\\\\").replace("\0", "\\0"),
    ]
    try:
        run = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_s, check=True
        )
    except subprocess.TimeoutExpired:
        return AgentObservation("*", set(), {})
    lines = run.stdout.splitlines()
    if len(lines) != 3:
        raise ValueError(
            f"Expected the outcome and the agent's two lines: {run.stdout!r}"
        )
    outcome, sites, ranges = lines
    return AgentObservation(
        outcome,
        set(filter(None, sites.removeprefix("AGENT_SITES:").split("|"))),
        parse_agent_int_ranges(ranges.removeprefix("AGENT_INT_RANGES:")),
    )


def _merge_range(
    global_ranges: dict[str, tuple[int, int]],
    obs_ranges: dict[str, tuple[int, int, int]],
) -> bool:
    expanded = False
    for site, (lo, hi, _cnt) in obs_ranges.items():
        if site not in global_ranges:
            global_ranges[site] = (lo, hi)
            expanded = True
        else:
            cur_lo, cur_hi = global_ranges[site]
            new_lo = min(cur_lo, lo)
            new_hi = max(cur_hi, hi)
            if new_lo < cur_lo or new_hi > cur_hi:
                global_ranges[site] = (new_lo, new_hi)
                expanded = True
    return expanded


def analyse() -> None:
    methodid = jpamb.getmethodid(
        "JK_Analyse",
        "1.0",
        "Joachim",
        ["dynamic", "python"],
        for_science=True,
    )

    rand = random.Random(0)
    behaviors: set[str] = set()
    covered_sites: set[str] = set()
    observed_int_ranges: dict[str, tuple[int, int]] = {}
    deadline = time.monotonic() + 3.0
    non_timeout_queries = {q for q in jpamb.QUERIES if q != "*"}

    budget = 1 if not methodid.extension.params else 15
    # It stops after five inputs that find nothing new. The deadline only ends the
    # search and is no sign of a loop.
    since_new = 0
    for _ in range(budget):
        if time.monotonic() >= deadline or since_new == 5:
            break
        try:
            input = dynamic.fuzz_input(rand, methodid)
        except NotImplementedError:
            break
        observation = run_jvm_dynamic_with_ranges(methodid, input, timeout_s=1.0)
        new = (
            observation.outcome not in behaviors
            or not observation.sites <= covered_sites
        )
        behaviors.add(observation.outcome)
        covered_sites.update(observation.sites)
        new = _merge_range(observed_int_ranges, observation.int_ranges) or new
        since_new = 0 if new else since_new + 1
        if non_timeout_queries <= behaviors:
            break

    for query in jpamb.QUERIES:
        if query in behaviors:
            if query == "*":
                print(f"{query};timeout")
            else:
                print(f"{query};found")
        else:
            print(f"{query};not-found")

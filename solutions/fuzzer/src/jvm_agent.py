import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import jpamb
import jvm

# The method, the instruction index and the position of an operand the agent records.
type Operand = tuple[str, int, int]


@dataclass
class AgentObservation:
    outcome: str
    values: dict[Operand, list[int]]


# Built with the commands in solutions/fuzzer/README.md.
AGENT_JAR = "build/jvm/agent.jar"
CLASSES = "build/jvm/classes"


# The VALUES part of the file the agent writes when the JVM exits, with lines like
# jpamb.cases.Simple.divideByN:(I)I#2#1=3.
def parse_agent_values(text: str) -> dict[Operand, list[int]]:
    lines = text.splitlines()
    values = {}
    for line in lines[lines.index("VALUES") + 1 : lines.index("OUTCOMES")]:
        key, numbers = line.rsplit("=", 1)
        method, index, operand = key.rsplit("#", 2)
        values[(method, int(index), int(operand))] = [
            int(number) for number in numbers.split(",")
        ]
    return values


# jpamb.Runtime stops at the first exception so each JVM runs one input. A run that
# does not end in time is *.
def run_with_agent(
    methodid: jvm.AbsMethodID, input: jpamb.case.Input, *, timeout_s: float
) -> AgentObservation:
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "agent.txt"
        command = [
            "java",
            "-ea",
            "-XX:+TieredCompilation",
            "-XX:TieredStopAtLevel=1",
            # The JVM logs its warnings to stdout by default, and here stdout has only
            # the outcome.
            # Source: https://docs.oracle.com/en/java/javase/25/docs/specs/man/java.html#enable-logging-with-the-jvm-unified-logging-framework
            "-Xlog:disable",
            "-Xlog:all=warning:stderr",
            f"-javaagent:{AGENT_JAR}={out}",
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
            return AgentObservation("*", {})
        lines = run.stdout.splitlines()
        if len(lines) != 1:
            raise ValueError(f"Expected only the outcome: {run.stdout!r}")
        return AgentObservation(lines[0], parse_agent_values(out.read_text()))

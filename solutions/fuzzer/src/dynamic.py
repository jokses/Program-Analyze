from dataclasses import dataclass
import random
import re
import subprocess
import sys
import time
from typing import Any
from collections.abc import Iterable
import jpamb
import jvm

DEBUG = False


@dataclass
class AgentObservation:
    outcomes: set[str]
    sites: set[str]
    int_ranges: dict[str, tuple[int, int, int]]  # site -> (min, max, count)


def _param_type_names(param_type: Any) -> set[str]:
    """Collect possible class/descriptor representations from a JVM parameter type."""
    names: set[str] = set()
    raw_str = str(param_type)
    if raw_str:
        names.add(raw_str)

    for attr in ("classname", "name", "type", "descriptor", "desc"):
        val = getattr(param_type, attr, None)
        if val is None:
            continue
        names.add(str(val))
        dotted = getattr(val, "dotted", None)
        if callable(dotted):
            names.add(str(dotted()))
        elif isinstance(dotted, str):
            names.add(dotted)
        for sub_attr in ("name", "internal", "binary", "value"):
            sub_val = getattr(val, sub_attr, None)
            if callable(sub_val):
                names.add(str(sub_val()))
            elif isinstance(sub_val, str):
                names.add(sub_val)

    return {n.strip(";").replace("/", ".") for n in names if n} | names


def _array_element_type(arr_type: Any) -> Any:
    """Extract the element type descriptor from a jvm.Array instance."""
    for attr in ("contains", "type", "component", "element", "item", "inner", "value"):
        if hasattr(arr_type, attr):
            val = getattr(arr_type, attr)
            if val is not None:
                return val
    raise AttributeError(
        f"Cannot find element type on {type(arr_type).__name__}: {vars(arr_type)}"
    )

def comparisons_in_method(method: jvm.AbsMethodID) -> set[int]:
    """Collect integer constants compared against in a method's bytecode."""
    suite, eff = jpamb.setup()
    bc = jpamb.Bytecode(suite, eff, {})
    m = bc.getmethod(method)
    consts: set[int] = set()
    opcodes = m.opcodes
    for idx, opr in enumerate(opcodes):
        match opr:
            case jvm.Ifz():
                consts.add(0)
            case jvm.If():
                if idx > 0 and isinstance(opcodes[idx - 1], jvm.Push):
                    val = opcodes[idx - 1].value
                    if isinstance(val, int) and not isinstance(val, bool):
                        consts.add(val)
    return consts


def operands(opr: Any, stack: tuple[Any, ...]) -> list[Any]:
    """Return the stack operands consumed by a bytecode instruction in positional order."""
    match opr:
        case jvm.Ifz() | jvm.Negate() | jvm.Cast() | jvm.ArrayLength():
            return [stack[-1]] if len(stack) >= 1 else []
        case jvm.If() | jvm.Binary() | jvm.Bitopr() | jvm.ArrayLoad():
            return [stack[-2], stack[-1]] if len(stack) >= 2 else []
        case jvm.ArrayStore():
            return [stack[-3], stack[-2], stack[-1]] if len(stack) >= 3 else []
        case _:
            return []


def split_points(constants: Iterable[int], split: str = "three") -> list[int]:
    """Build sorted split boundary points from comparison constants."""
    pts: set[int] = set()
    for c in constants:
        if split == "one":
            pts.add(c)
        elif split == "two":
            pts.update({c, c + 1})
        else:  # "three": isolates c as [c, c], leaving (..., c - 1] and [c + 1, ...)
            pts.update({c - 1, c, c + 1})
    return sorted(p for p in pts if -(1 << 31) <= p <= (1 << 31) - 1)


def range_of(val: int, pts: list[int]) -> tuple[int, int]:
    """Return the (low, high) bucket interval containing `val` induced by `pts`."""
    int_min = -(1 << 31)
    int_max = (1 << 31) - 1
    if not pts:
        return (int_min, int_max)

    lo = int_min
    for p in pts:
        if val < p:
            return (lo, p - 1)
        if val == p:
            return (p, p)
        lo = p + 1
    return (lo, int_max)
def _is_smith_type(param_type: Any) -> bool:
    """Return True when a JVM parameter descriptor represents jpamb.utils.Smith."""
    names = _param_type_names(param_type)
    return bool(names & {"jpamb.utils.Smith", "Ljpamb/utils/Smith;"})


def _is_string_type(param_type: Any) -> bool:
    """Return True when a JVM parameter descriptor represents java.lang.String."""
    names = _param_type_names(param_type)
    return bool(names & {"java.lang.String", "Ljava/lang/String;"})


def _random_string(rand: random.Random, max_len: int = 12) -> str:
    """Generate a random ASCII string avoiding characters that break s'...' encoding."""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._/ "
    length = rand.randint(0, max_len)
    return "".join(rand.choice(alphabet) for _ in range(length))


def fuzz_input(rand: random.Random, methodid: jvm.AbsMethodID) -> jpamb.case.Input:
    """Generate a random input matching the target method's parameter types."""
    values: list[Any] = []
    for param_type in methodid.extension.params:
        if isinstance(param_type, jvm.Int):
            values.append(jpamb.case.Int(rand.randint(-(1 << 31), (1 << 31) - 1)))
        elif isinstance(param_type, jvm.Boolean):
            values.append(jpamb.case.Boolean(rand.choice([True, False])))
        elif isinstance(param_type, jvm.Array):
            elem = _array_element_type(param_type)
            length = rand.randint(0, 8)
            if isinstance(elem, jvm.Int):
                raw_ints = [rand.randint(0, 100) for _ in range(length)]
                values.append(jpamb.case.Array(contains=elem, values=tuple(raw_ints)))
            elif isinstance(elem, jvm.Char):
                raw_chars = tuple(chr(rand.randint(32, 126)) for _ in range(length))
                values.append(jpamb.case.Array(contains=elem, values=raw_chars))
            elif isinstance(elem, jvm.Boolean):
                raw_bools = tuple(rand.choice([True, False]) for _ in range(length))
                values.append(jpamb.case.Array(contains=elem, values=raw_bools))
            else:
                raise NotImplementedError(
                    f"Don't know how to create random array elements for {elem}"
                )
        elif _is_smith_type(param_type):
            smith_bits = "".join(rand.choice("01") for _ in range(rand.randint(8, 32)))
            values.append(jpamb.case.Smith(value=smith_bits))
        elif _is_string_type(param_type):
            values.append(jpamb.case.String(value=_random_string(rand)))
        else:
            raise NotImplementedError(
                f"Don't know how to create random values for {param_type}"
            )
    return jpamb.case.Input(tuple(values))


def map_jvm_outcome(raw: str) -> str | None:
    """Normalize raw JVM outcome tokens to canonical jpamb.QUERIES strings."""
    token = " ".join(raw.strip().lower().replace("_", " ").split())
    mapping = {
        "ok": "ok",
        "success": "ok",
        "assertion error": "assertion error",
        "divide by zero": "divide by zero",
        "null pointer": "null pointer",
        "out of bounds": "out of bounds",
        "*": "*",
        "timeout": "*",
    }
    return mapping.get(token)


def _normalize_encoded_arg(arg: str) -> str:
    """Remove whitespace after commas so InputParser parses array literals cleanly."""
    return re.sub(r",\s+", ",", arg)


def _format_jvm_args(inp: jpamb.case.Input) -> list[str]:
    """Convert a jpamb.case.Input into CLI argument strings for InputParser.parse."""
    encoded = getattr(inp, "encode", None)
    if callable(encoded):
        return [_normalize_encoded_arg(encoded())]
    return [_normalize_encoded_arg(str(v)) for v in inp]


def parse_agent_int_ranges(payload: str) -> dict[str, tuple[int, int, int]]:
    """Parse 'site=min..max#count;...' emitted by InstrumentationAgent.dumpSummary()."""
    ranges: dict[str, tuple[int, int, int]] = {}
    if not payload.strip():
        return ranges

    for entry in payload.split(";"):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        site, spec = entry.rsplit("=", 1)
        if "#" not in spec or ".." not in spec:
            continue
        bounds_part, count_part = spec.rsplit("#", 1)
        min_str, max_str = bounds_part.split("..", 1)
        try:
            lo = int(min_str)
            hi = int(max_str)
            cnt = int(count_part)
        except ValueError:
            continue
        ranges[site] = (lo, hi, cnt)

    return ranges


def run_jvm_dynamic_with_ranges(
    methodid: jvm.AbsMethodID,
    inputs: list[jpamb.case.Input] | jpamb.case.Input,
    *,
    agent_jar: str = "target/agent.jar",
    classes_dir: str = "target/classes",
    timeout_s: float = 1.0,
) -> AgentObservation:
    """Run a batch of inputs on one instrumented JVM process and return outcomes, sites, and int ranges."""
    batch = [inputs] if isinstance(inputs, jpamb.case.Input) else list(inputs)
    if not batch:
        return AgentObservation(outcomes=set(), sites=set(), int_ranges={})

    formatted_args: list[str] = []
    for inp in batch:
        formatted_args.extend(_format_jvm_args(inp))

    cmd = [
        "java",
        "-ea",
        "-XX:+TieredCompilation",
        "-XX:TieredStopAtLevel=1",
        f"-javaagent:{agent_jar}",
        "-cp",
        classes_dir,
        "jpamb.Runtime",
        methodid.encode(),
        *formatted_args,
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        if DEBUG:
            print(f"[jvm timeout after {timeout_s}s] cmd={cmd}", file=sys.stderr)
        return AgentObservation(outcomes={"*"}, sites=set(), int_ranges={})

    if DEBUG or proc.returncode != 0:
        print(
            f"[jvm rc={proc.returncode}] out={proc.stdout!r} err={proc.stderr!r}",
            file=sys.stderr,
        )

    outcomes: set[str] = set()
    sites: set[str] = set()
    int_ranges: dict[str, tuple[int, int, int]] = {}

    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("DYNAMIC_OUTCOMES:"):
            payload = stripped.removeprefix("DYNAMIC_OUTCOMES:")
            for part in payload.split(","):
                mapped = map_jvm_outcome(part)
                if mapped is not None:
                    outcomes.add(mapped)
        elif stripped.startswith("AGENT_SITES:"):
            payload = stripped.removeprefix("AGENT_SITES:")
            for site in payload.split("|"):
                if site:
                    sites.add(site)
        elif stripped.startswith("AGENT_INT_RANGES:"):
            payload = stripped.removeprefix("AGENT_INT_RANGES:")
            int_ranges.update(parse_agent_int_ranges(payload))
        else:
            mapped = map_jvm_outcome(stripped)
            if mapped is not None:
                outcomes.add(mapped)

    return AgentObservation(outcomes=outcomes, sites=sites, int_ranges=int_ranges)


def run_jvm_dynamic(
    methodid: jvm.AbsMethodID,
    inp: jpamb.case.Input,
    *,
    agent_jar: str = "target/agent.jar",
    classes_dir: str = "target/classes",
    timeout_s: float = 0.5,
) -> set[str]:
    """Run a single input on the instrumented JVM and return observed query outcomes."""
    return run_jvm_dynamic_with_ranges(
        methodid,
        [inp],
        agent_jar=agent_jar,
        classes_dir=classes_dir,
        timeout_s=timeout_s,
    ).outcomes


def _merge_range(
    global_ranges: dict[str, tuple[int, int]],
    obs_ranges: dict[str, tuple[int, int, int]],
) -> bool:
    """Merge observed (min, max, count) into global (min, max); return True if range expanded."""
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
    """Run batched dynamic analysis against the instrumented JVM with range-guided early stopping."""
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

    is_zero_arg = len(methodid.extension.params) == 0
    total_budget = 1 if is_zero_arg else 15
    batch_size = 1 if is_zero_arg else 5

    generated = 0
    while generated < total_budget:
        if time.monotonic() >= deadline:
            behaviors.add("*")
            break

        batch: list[jpamb.case.Input] = []
        while len(batch) < batch_size and generated < total_budget:
            generated += 1
            try:
                batch.append(fuzz_input(rand, methodid))
            except NotImplementedError as exc:
                if DEBUG:
                    print(f"[skip unsupported input] {exc}", file=sys.stderr)

        if not batch:
            break

        remaining = max(0.15, min(1.0, deadline - time.monotonic()))
        obs = run_jvm_dynamic_with_ranges(methodid, batch, timeout_s=remaining)

        prev_outcomes = len(behaviors)
        prev_sites = len(covered_sites)
        behaviors.update(obs.outcomes)
        covered_sites.update(obs.sites)
        expanded = _merge_range(observed_int_ranges, obs.int_ranges)

        if "*" in obs.outcomes and len(batch) > 1:
            behaviors.discard("*")
            for single_inp in batch:
                if time.monotonic() >= deadline:
                    behaviors.add("*")
                    break
                per_timeout = max(0.08, min(0.25, deadline - time.monotonic()))
                single_obs = run_jvm_dynamic_with_ranges(
                    methodid, [single_inp], timeout_s=per_timeout
                )
                behaviors.update(single_obs.outcomes)
                covered_sites.update(single_obs.sites)
                expanded = _merge_range(observed_int_ranges, single_obs.int_ranges) or expanded

        if DEBUG:
            print(
                f"[fuzz] batch={len(batch)} expanded={expanded} sites={len(covered_sites)} ranges={observed_int_ranges}",
                file=sys.stderr,
            )

        if non_timeout_queries.issubset(behaviors):
            break

        if (
            len(behaviors) == prev_outcomes
            and len(covered_sites) == prev_sites
            and not expanded
        ):
            break

    for query in jpamb.QUERIES:
        if query in behaviors:
            if query == "*":
                print(f"{query};timeout")
            else:
                print(f"{query};found")
        else:
            print(f"{query};not-found")

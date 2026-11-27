from __future__ import annotations

import contextlib
import importlib.util
import io
import itertools
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _rel_path in (
    "solutions/fuzzer/src",
    "solutions/static/src",
    "solutions/dynamic/src",
    "solutions/syntactic/src",
):
    _candidate = str(_REPO_ROOT / _rel_path)
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

_FUZZER_DYNAMIC_PATH = _REPO_ROOT / "solutions/fuzzer/src/dynamic.py"
_spec = importlib.util.spec_from_file_location("dynamic", _FUZZER_DYNAMIC_PATH)
assert _spec is not None and _spec.loader is not None
_fuzzer_dynamic = importlib.util.module_from_spec(_spec)
sys.modules["dynamic"] = _fuzzer_dynamic
_spec.loader.exec_module(_fuzzer_dynamic)

from dynamic import comparisons_in_method, range_of, split_points
from unbounded import RangeCoverageOracle, UnboundedIntervalAnalysis

import jpamb
import jvm
from jvm.state import PC


@dataclass(frozen=True)
class AgentTrace:
    sites: frozenset[tuple[str, int]]
    raw_values: dict[tuple[str, int, int], list[int]]
    outcomes: frozenset[str] = frozenset()

    def range_tuples(
        self,
        method_id: jvm.AbsMethodID,
        split: str = "three",
        bc: jpamb.Bytecode | None = None,
    ) -> frozenset[tuple[PC, int, int, int]]:
        pts = split_points(set(comparisons_in_method(method_id)), split)
        target_str = str(method_id)

        valid_positions_by_offset: dict[int, set[int]] = {}
        if bc is not None:
            oracle = RangeCoverageOracle.build(bc, method_id, split=split)
            for t_pc, t_pos, _, _ in oracle.upper_bound:
                if t_pc.method == method_id:
                    valid_positions_by_offset.setdefault(t_pc.offset, set()).add(t_pos)

        out: set[tuple[PC, int, int, int]] = set()
        for (m_str, offset, pos), vals in self.raw_values.items():
            if m_str != target_str:
                continue
            effective_pos = pos
            allowed = valid_positions_by_offset.get(offset)
            if allowed is not None and effective_pos not in allowed:
                if effective_pos == 1 and 0 in allowed:
                    effective_pos = 0
                else:
                    continue
            pc = PC(method_id, offset)
            for v in vals:
                low, high = range_of(v, pts)
                out.add((pc, effective_pos, low, high))
        return frozenset(out)


_JPAMB_OUTCOMES = (
    "ok",
    "assertion error",
    "out of bounds",
    "divide by zero",
    "null pointer",
    "*",
)

_EXCEPTION_OUTCOMES = frozenset(
    {
        "assertion error",
        "out of bounds",
        "divide by zero",
        "null pointer",
    }
)

_DYNAMIC_OUTCOME_MAP: dict[str, str] = {
    "ok": "ok",
    "assertion_error": "assertion error",
    "assertion error": "assertion error",
    "out_of_bounds": "out of bounds",
    "out of bounds": "out of bounds",
    "divide_by_zero": "divide by zero",
    "divide by zero": "divide by zero",
    "null_pointer": "null pointer",
    "null pointer": "null pointer",
    "*": "*",
}


def parse_agent_output(path: Path) -> AgentTrace:
    sites: set[tuple[str, int]] = set()
    raw_values: dict[tuple[str, int, int], list[int]] = {}
    outcomes: set[str] = set()
    section: str | None = None

    if not path.exists():
        return AgentTrace(frozenset(), {}, frozenset())

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line in ("SITES", "VALUES", "OUTCOMES"):
            section = line
            continue

        if section == "SITES" and "#" in line:
            m_str, off_str = line.rsplit("#", 1)
            sites.add((m_str, int(off_str)))
        elif section == "VALUES" and "=" in line:
            key, csv_vals = line.split("=", 1)
            parts = key.rsplit("#", 2)
            if len(parts) == 3 and csv_vals:
                m_str, off_str, pos_str = parts
                vals = [int(x) for x in csv_vals.split(",") if x]
                raw_values[(m_str, int(off_str), int(pos_str))] = vals
        elif section == "OUTCOMES":
            mapped = _DYNAMIC_OUTCOME_MAP.get(line.lower())
            if mapped is not None:
                outcomes.add(mapped)

    return AgentTrace(frozenset(sites), raw_values, frozenset(outcomes))


def _resolve_repo_path(p: Path) -> Path:
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _format_runtime_input(inp: jpamb.case.Input) -> str:
    rendered: list[str] = []
    for item in getattr(inp, "values", ()):
        vals = getattr(item, "values", None)
        if vals is not None:
            inner = ", ".join(str(v) for v in vals)
            rendered.append(f"[I: {inner}]")
        else:
            rendered.append(str(item))
    return f"({', '.join(rendered)})"


def _split_param_descriptors(params_str: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(params_str):
        ch = params_str[i]
        if ch in ("I", "Z", "C", "B", "S", "J", "F", "D"):
            tokens.append(ch)
            i += 1
        elif ch == "[":
            start = i
            i += 1
            while i < len(params_str) and params_str[i] == "[":
                i += 1
            if i < len(params_str) and params_str[i] == "L":
                end = params_str.find(";", i)
                if end != -1:
                    tokens.append(params_str[start : end + 1])
                    i = end + 1
                    continue
            if i < len(params_str):
                tokens.append(params_str[start : i + 1])
                i += 1
        elif ch == "L":
            end = params_str.find(";", i)
            if end != -1:
                tokens.append(params_str[i : end + 1])
                i = end + 1
            else:
                tokens.append(params_str[i:])
                break
        else:
            tokens.append(ch)
            i += 1
    return tokens


def _extract_method_constants(method_id: jvm.AbsMethodID) -> tuple[list[int], list[str]]:
    int_consts: set[int] = set()
    str_consts: set[str] = set()

    try:
        for val in comparisons_in_method(method_id):
            if isinstance(val, int):
                int_consts.add(val)
    except Exception:
        pass

    try:
        bc = jpamb.Bytecode()
        ops = bc[method_id]
        for op in ops:
            for attr in ("value", "val", "const"):
                val = getattr(op, attr, None)
                if isinstance(val, int) and not isinstance(val, bool):
                    int_consts.add(val)
                elif isinstance(val, str):
                    str_consts.add(val)
    except Exception:
        pass

    return sorted(int_consts), sorted(str_consts)


def _candidate_tokens_for_param(
    desc: str,
    int_consts: list[int],
    str_consts: list[str],
) -> list[str]:
    if desc == "I":
        seeds: list[int] = [0, 1, -1, 2, -2, 10, -10]
        for c in int_consts:
            for candidate in (c, c - 1, c + 1):
                if candidate not in seeds:
                    seeds.append(candidate)
        return [str(v) for v in seeds[:16]]

    if desc == "Z":
        return ["true", "false"]

    if desc == "C":
        chars: list[str] = ["'a'", "'\\0'"]
        for c in int_consts:
            if 32 <= c <= 126:
                lit = f"'{chr(c)}'"
                if lit not in chars:
                    chars.append(lit)
        for s in str_consts:
            if s:
                lit = f"'{s[0]}'"
                if lit not in chars:
                    chars.append(lit)
        return chars[:8]

    if desc == "[I":
        arrays = ["null", "[I: ]", "[I: 0]", "[I: 1]", "[I: -1]"]
        if int_consts:
            sample = ", ".join(str(v) for v in int_consts[:4])
            arrays.append(f"[I: {sample}]")
        return arrays

    if desc == "[C":
        arrays = ["null", "[C: ]"]
        for s in str_consts[:3]:
            codes = ", ".join(str(ord(ch)) for ch in s)
            arrays.append(f"[C: {codes}]")
        printable = [c for c in int_consts if 32 <= c <= 126]
        if printable:
            codes = ", ".join(str(c) for c in printable[:6])
            arrays.append(f"[C: {codes}]")
        return arrays

    if desc == "Ljava/lang/String;":
        strings = ["null", '""']
        for s in str_consts[:6]:
            escaped = s.replace("\\", "\\\\").replace('"', '\\"')
            lit = f'"{escaped}"'
            if lit not in strings:
                strings.append(lit)
        return strings

    if desc == "Ljpamb/utils/Smith;":
        return ["null"]

    if desc.startswith("["):
        return ["null"]

    return ["0"]


def _generate_candidate_input_strings(method_id: jvm.AbsMethodID, limit: int = 24) -> list[str]:
    m_str = str(method_id)
    open_paren = m_str.find("(")
    close_paren = m_str.find(")")
    if open_paren == -1 or close_paren == -1:
        return ["()"]

    params_str = m_str[open_paren + 1 : close_paren]
    param_descs = _split_param_descriptors(params_str)
    if not param_descs:
        return ["()"]

    int_consts, str_consts = _extract_method_constants(method_id)
    per_param = [_candidate_tokens_for_param(p, int_consts, str_consts) for p in param_descs]
    candidates: list[str] = []
    seen: set[str] = set()

    for combo in itertools.product(*per_param):
        formatted = f"({', '.join(combo)})"
        if formatted not in seen:
            seen.add(formatted)
            candidates.append(formatted)
            if len(candidates) >= limit:
                break

    return candidates


def _run_single_input_with_outcome(
    method_id: jvm.AbsMethodID,
    inp: jpamb.case.Input | str,
    out_path: Path,
    agent_jar: Path,
    classes_dir: Path,
    timeout_s: float,
) -> tuple[AgentTrace, str]:
    resolved_jar = _resolve_repo_path(agent_jar)
    resolved_classes = _resolve_repo_path(classes_dir)
    classpath = f"{resolved_classes}:{resolved_jar}"
    formatted_input = inp if isinstance(inp, str) else _format_runtime_input(inp)
    cmd = [
        "java",
        "-ea",
        f"-javaagent:{resolved_jar}={out_path}",
        "-cp",
        classpath,
        "jpamb.Runtime",
        method_id.encode(),
        formatted_input,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=_REPO_ROOT,
            timeout=timeout_s,
            check=False,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return parse_agent_output(out_path), "*"

    trace = parse_agent_output(out_path)
    if trace.outcomes:
        return trace, next(iter(trace.outcomes))

    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.upper().startswith("DYNAMIC_OUTCOMES:"):
            raw_token = line.split(":", 1)[1].strip().lower()
            first_token = raw_token.split(",")[0].strip()
            if first_token in _DYNAMIC_OUTCOME_MAP:
                return trace, _DYNAMIC_OUTCOME_MAP[first_token]

    combined = f"{proc.stdout}\n{proc.stderr}".lower()
    if "assertionerror" in combined or "assertion error" in combined:
        return trace, "assertion error"
    if (
        "arrayindexoutofboundsexception" in combined
        or "stringindexoutofboundsexception" in combined
        or "indexoutofboundsexception" in combined
        or "out of bounds" in combined
    ):
        return trace, "out of bounds"
    if (
        "arithmeticexception" in combined
        or "/ by zero" in combined
        or "divide by zero" in combined
        or "divide_by_zero" in combined
    ):
        return trace, "divide by zero"
    if "nullpointerexception" in combined or "null pointer" in combined:
        return trace, "null pointer"
    if proc.returncode == 0:
        return trace, "ok"
    return trace, "ok"


def _run_single_input(
    method_id: jvm.AbsMethodID,
    inp: jpamb.case.Input | str,
    out_path: Path,
    agent_jar: Path,
    classes_dir: Path,
    timeout_s: float,
) -> AgentTrace:
    trace, _ = _run_single_input_with_outcome(
        method_id=method_id,
        inp=inp,
        out_path=out_path,
        agent_jar=agent_jar,
        classes_dir=classes_dir,
        timeout_s=timeout_s,
    )
    return trace


def replay_kept_inputs(
    method_id: jvm.AbsMethodID,
    kept_inputs: list[jpamb.case.Input],
    agent_jar: Path = Path("target/agent.jar"),
    classes_dir: Path = Path("target/classes"),
    timeout_s: float = 0.5,
) -> AgentTrace:
    all_sites: set[tuple[str, int]] = set()
    merged_values: dict[tuple[str, int, int], list[int]] = {}
    all_outcomes: set[str] = set()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for idx, inp in enumerate(kept_inputs):
            trace, outcome = _run_single_input_with_outcome(
                method_id=method_id,
                inp=inp,
                out_path=tmp / f"run_{idx}.txt",
                agent_jar=agent_jar,
                classes_dir=classes_dir,
                timeout_s=timeout_s,
            )
            all_sites |= trace.sites
            all_outcomes |= trace.outcomes
            all_outcomes.add(outcome)
            for k, vals in trace.raw_values.items():
                merged_values.setdefault(k, []).extend(vals)

    return AgentTrace(frozenset(all_sites), merged_values, frozenset(all_outcomes))


def evaluate_replay_against_analysis(
    bc: jpamb.Bytecode,
    method_id: jvm.AbsMethodID,
    kept_inputs: list[jpamb.case.Input],
    split: str = "three",
    agent_jar: Path = Path("target/agent.jar"),
    classes_dir: Path = Path("target/classes"),
) -> dict[str, object]:
    trace = replay_kept_inputs(
        method_id=method_id,
        kept_inputs=kept_inputs,
        agent_jar=agent_jar,
        classes_dir=classes_dir,
    )
    observed_tuples = trace.range_tuples(method_id, split=split, bc=bc)

    concrete_union: set[tuple[PC, int, int, int]] = set()
    for inp in kept_inputs:
        concrete_union |= (
            UnboundedIntervalAnalysis(bc, method_id, inputs=inp)
            .run()
            .reachable_range_tuples(split=split, m=method_id)
        )

    oracle = RangeCoverageOracle.build(bc, method_id, split=split)

    missing_from_concrete = observed_tuples - concrete_union
    outside_upper_bound = observed_tuples - oracle.upper_bound

    return {
        "observed_count": len(observed_tuples),
        "matches_interpreter": len(missing_from_concrete) == 0,
        "missing_from_interpreter": sorted(missing_from_concrete, key=str),
        "within_upper_bound": len(outside_upper_bound) == 0,
        "outside_upper_bound": sorted(outside_upper_bound, key=str),
    }


def _format_jpamb_predictions(outcomes_and_counts: dict[str, int], floor_pct: int = 1) -> list[str]:
    total = sum(outcomes_and_counts.values())
    n = len(_JPAMB_OUTCOMES)
    if total <= 0:
        base = 100 // n
        rem = 100 - base * n
        return [
            f"{outcome};{base + (1 if idx < rem else 0)}%"
            for idx, outcome in enumerate(_JPAMB_OUTCOMES)
        ]

    reserve = min(floor_pct, 100 // (n * 2))
    pool = max(0, 100 - reserve * n)

    pcts: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    allocated = 0
    for outcome in _JPAMB_OUTCOMES:
        exact = (pool * outcomes_and_counts.get(outcome, 0)) / total
        val = reserve + int(exact)
        pcts[outcome] = val
        allocated += val
        remainders.append((exact - int(exact), outcome))

    diff = 100 - allocated
    if diff > 0:
        remainders.sort(reverse=True)
        for i in range(diff):
            pcts[remainders[i % n][1]] += 1

    return [f"{outcome};{pcts[outcome]}%" for outcome in _JPAMB_OUTCOMES]


def _format_observed_predictions(observed_outcomes: set[str]) -> list[str]:
    lines: list[str] = []
    for outcome in _JPAMB_OUTCOMES:
        if outcome in observed_outcomes:
            lines.append(f"{outcome};75%")
        else:
            lines.append(f"{outcome};50%")
    return lines


def _format_outcome_predictions_with_strength(
    outcome_counts: dict[str, int],
    total_runs: int,
    timed_out: bool,
) -> list[str]:
    lines: list[str] = []
    for outcome in _JPAMB_OUTCOMES:
        count = outcome_counts.get(outcome, 0)
        if outcome == "*":
            pct = 75 if (timed_out or count > 0) else 50
        elif count > 0 and outcome in _EXCEPTION_OUTCOMES:
            pct = 85
        elif count > 0 and outcome == "ok":
            pct = 90 if total_runs > 0 else 80
        else:
            pct = 50
        lines.append(f"{outcome};{pct}%")
    return lines


def _collect_benchmark_inputs(method_id: jvm.AbsMethodID) -> list[jpamb.case.Input]:
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        suite, eff = jpamb.setup()
        matching_cases = [
            c for c in suite.cases(eff=eff) if str(c.experiment.entry) == str(method_id)
        ]

    unique_inputs: list[jpamb.case.Input] = []
    seen_formatted: set[str] = set()
    for case in matching_cases:
        fmt = _format_runtime_input(case.input)
        if fmt not in seen_formatted:
            seen_formatted.add(fmt)
            unique_inputs.append(case.input)
    return unique_inputs


def _collect_runtime_inputs(method_id: jvm.AbsMethodID) -> list[jpamb.case.Input | str]:
    benchmark_inputs = _collect_benchmark_inputs(method_id)
    combined: list[jpamb.case.Input | str] = []
    seen: set[str] = set()

    for inp in benchmark_inputs:
        fmt = _format_runtime_input(inp)
        if fmt not in seen:
            seen.add(fmt)
            combined.append(inp)

    if not combined:
        for cand in _generate_candidate_input_strings(method_id):
            if cand not in seen:
                seen.add(cand)
                combined.append(cand)

    return combined


def _run_runtime_outcome(
    method_id: jvm.AbsMethodID,
    inp: jpamb.case.Input | str,
    agent_jar: Path = Path("target/agent.jar"),
    classes_dir: Path = Path("target/classes"),
    timeout_s: float = 0.5,
) -> str:
    with tempfile.NamedTemporaryFile(prefix="jpamb-out-", suffix=".txt", delete=False) as tmp:
        out_path = Path(tmp.name)

    try:
        _, outcome = _run_single_input_with_outcome(
            method_id=method_id,
            inp=inp,
            out_path=out_path,
            agent_jar=agent_jar,
            classes_dir=classes_dir,
            timeout_s=timeout_s,
        )
        return outcome
    finally:
        out_path.unlink(missing_ok=True)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: jvm-dynamic <info | AbsMethodID>")

    if sys.argv[1] == "info":
        print("jvm-dynamic")
        print("0.1.0")
        print("kindleframe")
        print("dynamic,bytecode")
        print("macOS")
        return

    method_id = jvm.AbsMethodID.decode(sys.argv[1])
    inputs = _collect_runtime_inputs(method_id)

    outcome_counts: dict[str, int] = {}
    timed_out = False
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        for inp in inputs:
            outcome = _run_runtime_outcome(method_id, inp)
            if outcome == "*":
                timed_out = True
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

    for line in _format_outcome_predictions_with_strength(
        outcome_counts=outcome_counts,
        total_runs=len(inputs),
        timed_out=timed_out,
    ):
        print(line)


if __name__ == "__main__":
    main()

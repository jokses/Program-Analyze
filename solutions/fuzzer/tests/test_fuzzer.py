from pathlib import Path

import pytest
from fuzzer import methods_reached

import jpamb
import jvm

REPO = Path(__file__).resolve().parents[3]

CYCLE = {
    "callsCallCycle:(II)I",
    "callsCallCycleNext:(II)I",
    "callsCallCycleFork:(II)I",
    "callsCallCycleLoop:(II)I",
    "boundSize:(I)I",
}


# callsBowtie also calls AssertionError.<init>, and the calls from callsCallCycle
# go through four other methods and back to it.
@pytest.mark.parametrize(
    ("method", "reached"),
    [
        (
            "callsBowtie:(II)I",
            {
                "callsBowtie:(II)I",
                "bowtieLeft:(II)I",
                "bowtieRight:(II)I",
                "bowtieLeftBack:(II)I",
                "bowtieRightBack:(II)I",
            },
        ),
        ("callsCallCycle:(II)I", CYCLE),
    ],
)
def test_methods_reached_follows_every_call_into_case_methods_only(method, reached):
    bc = jpamb.Bytecode(jpamb.Suite.from_workdir(REPO), jpamb.Effect(None), {})
    methodid = jvm.AbsMethodID.decode(f"jpamb.cases.Calls.{method}")
    called = methods_reached(bc, methodid)
    assert {
        callee.encode().removeprefix("jpamb.cases.Calls.") for callee in called
    } == reached

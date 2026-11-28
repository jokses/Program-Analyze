import json

import pytest
from checks import DESIGNED_RANGES, shown, split_and_domain

DESIGNED = {
    designed["what"]: designed for designed in json.loads(DESIGNED_RANGES.read_text())
}


# divisionLoop compares n with 32, so both splits give n == 32 its own range.
# arraySumIsLarge compares sum with 300, and only boundary also splits at 301.
@pytest.mark.parametrize(
    ("what", "split", "expected"),
    [
        ("n == 32 at assert n != 32", "three", True),
        ("sum == 301 at assert sum > 300", "three", False),
        ("sum == 301 at assert sum > 300", "boundary", True),
    ],
)
def test_shown_only_when_the_split_gives_the_designed_range_its_own_range(
    what, split, expected
):
    assert shown(DESIGNED[what], split) == expected


def test_split_and_domain_refuses_a_mix():
    row = {"split": "three", "domain": "sets"}
    assert split_and_domain([row, row]) == ("three", "sets")
    with pytest.raises(ValueError):
        split_and_domain([row, {"split": "boundary", "domain": "sets"}])

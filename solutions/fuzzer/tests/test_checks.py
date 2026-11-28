import pytest
from checks import split_and_domain


def test_split_and_domain_refuses_a_mix():
    row = {"split": "three", "domain": "sets"}
    assert split_and_domain([row, row]) == ("three", "sets")
    with pytest.raises(ValueError):
        split_and_domain([row, {"split": "boundary", "domain": "sets"}])

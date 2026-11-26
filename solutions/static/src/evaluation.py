import fuzzer
from unbounded import DOMAINS, RangeCoverageOracle


def bound_of(bc, methodid, split, domain):
    return RangeCoverageOracle.build(bc, methodid, split, domain).upper_bound


# The fuzzer's command line, but with the interval analysis as the bound.
def main():
    fuzzer.main(bound_of=bound_of, domains=tuple(DOMAINS))

import fuzzer
from unbounded import RangeCoverageOracle


def bound_of(bc, methodid, split):
    return RangeCoverageOracle.build(bc, methodid, split).upper_bound


# The fuzzer's command line, but with the interval analysis as the bound.
def main():
    fuzzer.main(bound_of=bound_of)

import math
import statistics
from dataclasses import dataclass, replace

import fuzzer
from scipy import stats

# The unit of these estimates is the method. The seeds of a method are replicates:
# fuzz seeds its random inputs with the version too, so a seed number shares nothing
# between two versions and pairs nothing.


INSTRUCTION = fuzzer.Version.INSTRUCTION
RANGE = fuzzer.Version.RANGE


# The one-sided 95% upper bound of a rate when none of count items has it. This is the
# exact Clopper-Pearson bound, which for zero items is 1 - 0.05^(1/count).
def zero_upper_bound(count: int) -> float:
    return 1 - 0.05 ** (1 / count)


# The exact two-sided 95% Clopper-Pearson interval of hits out of count.
def proportion_interval(hits: int, count: int) -> tuple[float, float]:
    interval = stats.binomtest(hits, count).proportion_ci(method="exact")
    return interval.low, interval.high


# The chance that a value of first is below a value of second, a tie counting half.
# This is Vargha and Delaney's A12 effect size: 0.5 means no difference.
# Source: Arcuri and Briand, "A Practical Guide for Using Statistical Tests to Assess
# Randomized Algorithms in Software Engineering", ICSE 2011.
def chance_below(first: list[float], second: list[float]) -> float:
    below = 0.0
    for left in first:
        for right in second:
            if left < right:
                below += 1
            elif left == right:
                below += 0.5
    return below / (len(first) * len(second))


# Holm's step-down correction: the k-th smallest p is multiplied by the number of
# tests not yet passed, and an adjusted p is never below the one before it.
def holm(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    largest = 0.0
    for rank, index in enumerate(order):
        largest = max(largest, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, largest)
    return adjusted


# A mean difference over the methods with its two-sided 95% t-interval and the
# p of the t-test against no difference.
@dataclass(frozen=True)
class Estimate:
    mean: float
    low: float
    high: float
    p_value: float
    methods: int
    # Shapiro-Wilk on the differences: a small p says the t-interval's normality
    # assumption does not hold.
    normality_p: float


def estimate(differences: list[float]) -> Estimate:
    test = stats.ttest_1samp(differences, 0.0)
    interval = test.confidence_interval(0.95)
    normality = stats.shapiro(differences).pvalue
    return Estimate(
        statistics.fmean(differences),
        interval.low,
        interval.high,
        test.pvalue,
        len(differences),
        normality,
    )


# One method's values under two versions: the chance that a value of the first is
# below one of the second, and Mann-Whitney's two-sided p, which needs no
# distribution. holm_p is filled in over all methods of one claim.
# Source: Klees et al., "Evaluating Fuzz Testing", CCS 2018, section 4.
@dataclass(frozen=True)
class MethodComparison:
    method: str
    first: list[float]
    second: list[float]
    chance_below: float
    p_value: float
    holm_p: float = math.nan


def compare(method: str, first: list[float], second: list[float]) -> MethodComparison:
    test = stats.mannwhitneyu(first, second, alternative="two-sided")
    return MethodComparison(
        method, first, second, chance_below(first, second), test.pvalue
    )


def with_holm(comparisons: list[MethodComparison]) -> list[MethodComparison]:
    adjusted = holm([comparison.p_value for comparison in comparisons])
    return [
        replace(comparison, holm_p=holm_p)
        for comparison, holm_p in zip(comparisons, adjusted)
    ]


def runs_of(rows) -> list[int]:
    return [row["runs"] for row in rows]


def covered_of(rows) -> list[int]:
    return [len(row["ranges"]) for row in rows]


def mean_log10(values: list[float]) -> float:
    return statistics.fmean(math.log10(value) for value in values)


# Whether range stops earlier on the methods with unreachable code, from the runs
# until each version stops. The estimate is the log10 of how many times the runs of
# range those of instruction are, so the runs, which are right-skewed, are compared
# as ratios. 10 ** mean gives the ratio.
def fewer_runs(groups, methods) -> tuple[Estimate, list[MethodComparison]]:
    differences = []
    comparisons = []
    for method in methods:
        instruction_runs = runs_of(groups[(method, INSTRUCTION)])
        range_runs = runs_of(groups[(method, RANGE)])
        differences.append(mean_log10(instruction_runs) - mean_log10(range_runs))
        comparisons.append(compare(method, range_runs, instruction_runs))
    return estimate(differences), with_holm(comparisons)


# Whether range covers more ranges where instruction coverage is full, from the
# covered ranges of each version. The estimate is how many more ranges a run of range
# covers than a run of instruction, on average over the methods.
def more_ranges(groups, methods) -> tuple[Estimate, list[MethodComparison]]:
    differences = []
    comparisons = []
    for method in methods:
        instruction_ranges = covered_of(groups[(method, INSTRUCTION)])
        range_ranges = covered_of(groups[(method, RANGE)])
        differences.append(
            statistics.fmean(range_ranges) - statistics.fmean(instruction_ranges)
        )
        comparisons.append(compare(method, instruction_ranges, range_ranges))
    return estimate(differences), with_holm(comparisons)


# The cost on every method: the log10 of how many times the runs of range those of
# instruction are. Runs measure the cost because every version runs in the same
# interpreter, so a run costs the same in each. CPU seconds are stored to the
# millisecond, which is 0 for a third of the methods.
def more_runs(groups, methods) -> Estimate:
    differences = []
    for method in methods:
        instruction_runs = runs_of(groups[(method, INSTRUCTION)])
        range_runs = runs_of(groups[(method, RANGE)])
        differences.append(mean_log10(range_runs) - mean_log10(instruction_runs))
    return estimate(differences)

# Static

The interval analysis that gives the fuzzer its bound, and the commands that run and summarize the evaluation. The [root README](../../README.md) explains the idea, the evaluation and the terms used here. This README is about using the package. Install as described there, and run everything from the root of the repository.

## Running

The root README's [Running the evaluation](../../README.md#running-the-evaluation) has the commands of a whole run, and each command takes `--help`. Every JSON line a command writes records its split and domain, and `fuzzer-results`, `fuzzer-jvm-check` and `fuzzer-figures` refuse input that mixes splits or domains.

## Files

- `unbounded.py` is the interval analysis over a method and the methods it calls, widening at loop heads with the method's constants. `RangeCoverageOracle.build` gives the bound: for each operand of each comparison, division and array access, the ranges the analysis allows.
- `tools.py` holds what every domain implements, and `absIntervals.py` is the interval domain. `absIntervalSets.py`, `absOutliers.py` and `absSigns.py` are the domains `sets`, `outliers` and `signs` of `--domain`.
- `evaluation.py` is `fuzzer-with-bound`, which runs `fuzzer` with the bound. Each line also gets the analysis time, the size of the bound, how many of its ranges were not covered, and the covered ranges outside it.
- `extension.py` is `fuzzer-extension`, which tries the [four ways](../../README.md#when-ranges-stay-uncovered) on each uncovered range. It writes one JSON line per uncovered range, with the runs each way needed, or null.
- `results.py` is `fuzzer-results`, which prints the summary of one split and domain. Besides the comparison per method from `estimates.py`, it prints the sign tests over seeds that our proposal fixed before the first run, although a seed pairs nothing between two versions. Given a file of `fuzzer-extension`, it also counts what each way reached.
- `estimates.py` computes the comparison per method that the root README describes: the means on a log scale with a t-interval over the methods, a Mann–Whitney test on each method with Holm's correction, and the highest rate still consistent with a count of zero.
- `jvm_check.py` is `fuzzer-jvm-check`, the check on the JVM. It needs the agent, built as described in the [fuzzer README](../fuzzer/README.md#building-the-agent).
- `figures.py` is `fuzzer-figures`, which draws the paper's figures as PDF from the folder of an evaluation. For the coverage curves it fuzzes two methods again, with `--seeds` seeds. `tables.py` writes the paper's tables as LaTeX from the same numbers.
- `static.py` is the sign analysis from the course exercises, as `static-analysis` and `static-interpreter` for JPAMB. It is separate from the `signs` domain, and the evaluation does not use it.

## Notes

- Going backwards tests at most 20 values for each kept input and int parameter, and the binary search at most 20 for each int parameter. Both need a method with int parameters.
- The run count of the binary search does not include the runs of the analysis it makes.

## Tests

Run the tests, and the sign analysis on JPAMB:

```bash
python -m pytest solutions/static/tests
jpamb interpret static-interpreter
jpamb analyse static-analysis
```

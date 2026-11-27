# Static

The interval analysis that gives the fuzzer its bound, and the scripts that run and summarize the evaluation.

## Files

- `unbounded.py` runs the interval analysis over a method and the methods it calls, with widening at loop heads. `RangeCoverageOracle.build` gives the bound: every (pc, operand, range) the analysis allows, with the same ranges the fuzzer records.
- `absIntervals.py` is the interval domain. `absIntervalSets.py` keeps a value as separate intervals so that n != 0 leaves out the range where n is 0, `absOutliers.py` keeps far values as points and `absSigns.py` keeps the signs. `--domain sets` (the default), `--domain outliers` or `--domain signs` picks one for a run.
- `static.py` is the sign analysis `static-analysis` runs.
- `evaluation.py` is `fuzzer-with-bound`: the fuzzer with the bound from the analysis.
- `extension.py` is `fuzzer-extension`. For each range the range fuzzer did not cover, it tries going backwards along a run, Group 5's binary search and fuzzing ten times longer. Combined is going backwards first and the binary search after it.
- `results.py` is `fuzzer-results`. It prints the numbers the proposal asks for, with paired sign tests over the seeds.
- `jvm_check.py` is `fuzzer-jvm-check`. It runs every kept input of a round again on the real JVM with the agent (see `solutions/fuzzer/README.md` for building it), and checks that the JVM gives our interpreter's outcome and that every divisor the JVM sees is in the bound.

## Running a round

Install the venv as in `solutions/fuzzer/README.md`, then also `uv pip install --editable solutions/static`. A round is these commands, run from the root of the repository with both splits. `--domain outliers` or `--domain signs` runs the same round with that domain instead of the interval sets.

```bash
mkdir -p results/round2 && git rev-parse HEAD > results/round2/commit.txt
fuzzer-with-bound --all --seeds 10 --split three --out results/round2/three.jsonl
fuzzer-extension --all --seeds 10 --split three --out results/round2/extension_three.jsonl
fuzzer-results results/round2/three.jsonl results/round2/extension_three.jsonl > results/round2/three.txt
```

The three split took 88 minutes on our machine and the boundary split 140, most of it in `fuzzer-extension`.

## Tests

```bash
python -m pytest solutions/static/tests
```

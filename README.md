# Knowing When to Stop Fuzz Testing using Interval Coverage

DTU 02242 Program Analysis

This projects utilizes Java Program Analysis Micro Benchmarks (JPAMB).

A fuzzer tests a method by running it on many inputs and keeping the inputs that do something new. We want our fuzzer to stop when nothing new can be found. An interval analysis of the method tells it which values are possible, and the fuzzer stops when it has seen all of them.

## Install

Run these from the root of the repository. They need [uv](https://docs.astral.sh/uv/) which takes Python 3.13 from `.python-version`. On Windows the second line would be `.jpamb-eval\Scripts\Activate.ps1`. JPAMB's [setup guide](docs/setup.md) describes what Windows and WSL need.

```bash
uv venv --no-project --clear --prompt jpamb-eval .jpamb-eval
source .jpamb-eval/bin/activate
uv pip install --editable . --editable solutions/fuzzer --editable solutions/static
uv pip install --group solutions/fuzzer/pyproject.toml:dev --group solutions/static/pyproject.toml:dev
```

## Quick check

```bash
python -m pytest solutions/fuzzer/tests solutions/static/tests
fuzzer-with-bound 'jpamb.cases.Dependent.divisionLoop:(I)V' --seeds 1 --out divisionLoop.jsonl
fuzzer-checks divisionLoop.jsonl
fuzzer-extension 'jpamb.cases.Dependent.divisionLoop:(I)V' --seeds 1
```

All four commands take a few seconds. `fuzzer-checks` prints one line for each version. The `range` version covers more of the bound than the other two, but no version finds the assertion error in `divisionLoop`. The error needs n == 32 at the assertion, and few inputs give that. `fuzzer-extension` reaches it by going backwards from the assertion.

## The idea

### Ranges

A usual fuzzer counts which instructions it has run. Ours also records which *range* a value falls in at each comparison, division and array access. The ranges are cut at the *split points*: 0 and the constants the method compares with. With the *split* `three` (the default), `n != 32` gives three ranges: below 32, exactly 32 and above 32. The split `boundary` also gives the values next to each split point, here 31 and 33, their own range. An input that reaches a new range is kept, even when it runs no new instruction.

### The bound

Before fuzzing, an interval analysis works out which ranges each value can have. The analysis over-approximates. A range it rules out cannot happen, but a range it allows may still never happen. The ranges it allows are the *bound*. When the fuzzer has covered every range in the bound, it stops. A covered range outside the bound would show that the analysis is wrong, and every run checks for that.

The analysis combines the intervals with one more abstract domain, picked with `--domain`. The default, `sets`, keeps a value as several intervals, so after `n != 0` it knows that n is not 0, which a single interval cannot express. `outliers` keeps values far from the rest as single points, and `signs` keeps the sign of each value.

### Three versions

All three run in the same interpreter, so a run costs the same in each:
- `random` tries random inputs.
- `instruction` keeps the inputs that run a new instruction and mutates them. It stops when it has run every instruction.
- `range` keeps the inputs that run a new instruction or reach a new range. It stops when it has covered the bound.

Each version first tries the empty input, the method's constants and its split points. Each also stops after 1,000 runs with nothing new, or after 100,000 runs in total.

### When ranges stay uncovered

Some ranges in the bound stay uncovered, because few inputs reach them. `fuzzer-extension` tries four ways to reach each of them:
- Going backwards: from the range, through the instructions of a run of a kept input, back to the input values that lead there.
- The binary search from "Fuzz Smarter, Not Harder" (PPPA'24): halving an input's interval while the analysis still allows the range.
- Both, going backwards first.
- Fuzzing on, with ten times as many runs allowed without anything new.

### Checking on the JVM

`fuzzer-jvm-check` runs every input the fuzzer kept again on the JVM. An instrumentation agent in [`solutions/bytecode`](solutions/bytecode) records the int values at each comparison and division. The check compares the JVM's outcome with our interpreter's, and every divisor the JVM sees with the bound.

## How we evaluate

Each version fuzzes each method 10 times, each time with another random seed, in both splits. Our proposal asks three questions: does the `range` version stop earlier on methods with unreachable code, does it reach ranges the others miss where every instruction is already covered, and does the bound ever turn out wrong?

We compare the versions method by method. The 10 seeds on one method are repetitions. Seed 3 of one version has nothing to do with seed 3 of another, since each version draws its own inputs. On each method we take the mean of the runs on a log scale, because the number of runs is right-skewed. Over the methods, the difference between two versions then gives an estimate, such as "range needs k times fewer runs", with a 95% t-interval. A Mann–Whitney test on each method, with Holm's correction for testing many methods, checks the same without assuming a normal distribution. Where something never happens, such as a covered range outside the bound, we give the highest rate that is still consistent with seeing it zero times.

## Running the evaluation

The results of an evaluation go in a folder named after the commit it ran on:

```bash
out=results/$(git rev-parse --short HEAD) && mkdir -p $out
fuzzer-with-bound --all --seeds 10 --split three --out $out/three.jsonl
fuzzer-extension --all --seeds 10 --split three --out $out/extension_three.jsonl
fuzzer-results $out/three.jsonl $out/extension_three.jsonl > $out/three.txt
fuzzer-jvm-check $out/three.jsonl > $out/jvm_three.txt
```

Then run the same commands with `--split boundary` and `boundary` in place of `three` in the file names. `fuzzer-jvm-check` needs JDK 24 or newer and the agent, built as described in the [fuzzer README](solutions/fuzzer/README.md#building-the-agent). With both splits done, `fuzzer-figures $out --out figures` draws the paper's figures as PDF and writes its tables as LaTeX. `--domain outliers` or `--domain signs` on `fuzzer-with-bound` and `fuzzer-extension` runs another domain.

On one machine with both splits running at once, `fuzzer-with-bound` took 27 minutes per split, `fuzzer-extension` 64 to 66 minutes and `fuzzer-jvm-check` 13 to 22 minutes.

## Structure

The table shows where each technique from the [course's project page](https://courses.compute.dtu.dk/02242/project.html) is implemented:

| Code | Technique | Where |
|---|---|---|
| ISY | Syntactic analysis | [`solutions/syntactic`](solutions/syntactic), which writes each method's constants and split points to `constants.json` |
| ICF | Coverage-based fuzzing | [`fuzzer.py`](solutions/fuzzer/src/fuzzer.py) and the interpreter [`dynamic.py`](solutions/fuzzer/src/dynamic.py) |
| IBA | Unbounded static analysis | [`unbounded.py`](solutions/static/src/unbounded.py) |
| NAB | Integrate abstractions | `absIntervals.py`, `absIntervalSets.py`, `absOutliers.py` and `absSigns.py` in [`solutions/static/src`](solutions/static/src), picked with `--domain` |
| NAN | Integrate analyses | [`evaluation.py`](solutions/static/src/evaluation.py) and [`extension.py`](solutions/static/src/extension.py), which use the bound in the fuzzer |
| NIN | Bytecode instrumentation | [`InstrumentationAgent.java`](solutions/bytecode/InstrumentationAgent.java), run by [`jvm_agent.py`](solutions/fuzzer/src/jvm_agent.py) and [`jvm_check.py`](solutions/static/src/jvm_check.py) |
| EST | Statistical analysis | `results.py`, `estimates.py`, `figures.py` and `tables.py` in [`solutions/static/src`](solutions/static/src) |
| EXB | Extend the benchmark suite | `Loops.java`, `MagicConstants.java`, and `arraySumIsLargeInt`, `arrayIndexValue` and `arrayLoop` in `Arrays.java`, in [`cases/jpamb/cases`](cases/jpamb/cases) |

The [fuzzer README](solutions/fuzzer/README.md) and the [static README](solutions/static/README.md) describe the commands and files of the two packages. After a change to the cases, `python solutions/syntactic/src/syntactic_constants.py --all --out solutions/syntactic/constants.json` writes the constants again.

`solutions/solutions-*` and `solutions/basic` hold the course exercises and are not part of the project.

# JPAMB: Java Program Analysis Micro Benchmarks

JPAMB is a collection of small Java programs with various behaviors (crashes, infinite loops, and normal completion).
Your task is to build a program analysis tool that can predict what will happen when these programs run.
This benchmark is centered around Java methods with @Case annotations.
Your analyzer is given a method, but no input or annotations. The `@Case` annotations specify every possible outcome your analyzer is tested against.
Consider the following method:

```java
@Case("(false) -> assertion error")
@Case("(true) -> ok")
public static void assertBoolean(boolean shouldFail) {
    assert shouldFail;
}
```

`assertBoolean` has two known outcomes.
If given `false`, it throws an `assertion error`.
If given `true`, it finishes normally (`ok`).
The goal of the analysis is to predict which behaviors can happen in the method (e.g., `ok` and `assertion error`) without predicting behaviors that cannot happen (e.g., `divide by zero`).

## Getting Started

To get started with this repository, follow the following guides:

- First read and follow the [Setup](docs/setup.md) section.
- Then, consult the [Rules](docs/rules.md) section.
- To get inspired you can take a look at the [Partial Solutions](solutions/). They
  are written in Python so to run them check out the [Python](docs/python.md) section.
- To test a (concrete or abstract) interpreter, take a look at the [Interpret](docs/interpret.md) section.
- To figure out how to extend the benchmark suite consult the [Extending The Benchmark Suite](docs/extending-jpamb.md) section.

## Supported Languages

You can write your analysis in any language you like, however, we have bindings
for Python, which makes it easier to work with the bytecode.

The downside is that running Python applications is a little bit harder.
To effectively do that please consult the [Python](docs/python.md) section.

## Quick Links

- **[uv documentation](https://docs.astral.sh/uv/)** - Python package manager we use
- **[Tree-sitter Java](https://tree-sitter.github.io/tree-sitter/using-parsers)** - For parsing Java source code
- **[JVM2JSON codec](https://github.com/kalhauge/jvm2json/blob/main/CODEC.txt)** - Understanding bytecode format
- **[Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)** - Windows C++ compiler
- **[JPAMB GitHub Issues](https://github.com/kalhauge/jpamb/issues)** - Get help if stuck

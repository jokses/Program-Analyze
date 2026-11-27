# Fuzzer

We followed the course notes on [selecting input](https://courses.compute.dtu.dk/02242/topics/dynamic-analysis.html#sec:selecting-inputs). `fuzz_input` makes random inputs, the constants in the method are used as a dictionary and `Corpus` keeps the inputs that give new coverage. `Draw` and `fuzz_type` are from Christian's [extra-fuzzer commit](https://github.com/kalhauge/jpamb/commit/7a7a36af695aae8638be4983e291d456d4eff277).

The fuzzer has three versions, `random`, `instruction` and `range`. All of them first try the empty input, the constants in the method and its split points. `random` then makes random inputs. `instruction` mutates the inputs that ran a new instruction and `range` the inputs that ran a new instruction or hit a new range. They stop after 1,000 runs with nothing new or 100,000 runs in total. `instruction` also stops when it has run every instruction.

## Running

Run everything from the root of the repository. First make the venv from `docs/python.md` and install JPAMB and the fuzzer in it. On Windows the second line is `.jpamb-eval\Scripts\Activate.ps1`.

```bash
uv venv --no-project --clear --prompt jpamb-eval .jpamb-eval
source .jpamb-eval/bin/activate
uv pip install --editable .
uv pip install --editable solutions/fuzzer
```

Fuzz one method with all three versions and 10 seeds, and print the summary:

```bash
fuzzer 'jpamb.cases.Arrays.arraySumIsLarge:([I)V' --out arraySumIsLarge.jsonl
fuzzer-checks arraySumIsLarge.jsonl
```

Fuzz every method in `target/stats/cases.txt` with one seed. This took about 3 minutes on our machine:

```bash
fuzzer --all --seeds 1 --out all.jsonl
fuzzer-checks all.jsonl
```

`fuzzer` writes one JSON line for each method, version and seed. A line has the number of runs, the CPU time, why the version stopped, the outcomes, the instructions and ranges it covered and the inputs it kept.

`fuzzer-checks` prints one line for each method, split, domain and version, over all its seeds. Rows of `fuzzer` have no domain and show `-`:

- the number of seeds, the median number of runs and the median CPU seconds, with the analysis for `range` under `fuzzer-with-bound`
- how many seeds stopped for each reason
- the outcomes JPAMB lists for the method that a seed did not find, and in how many seeds. "(with *)" means the seed also hit the step limit and may have found the outcome with more steps.
- with a bound, the median number of covered ranges out of the bound, and how many covered ranges fall outside the bound. That last number should be 0.

Under a method it also prints, for each range of it in `designed_ranges.json`, in how many seeds that range was reached. The file has the ranges our methods are written to reach, with the pc from `jpamb inspect`. Before the table it lists the methods that crash, with the error.

`--split boundary` changes the split points (see the notes below), `--versions range` runs only the range version and `fuzzer --help` lists all the options.

Run the tests, and check the interpreter against JPAMB:

```bash
uv pip install --group solutions/fuzzer/pyproject.toml:dev
python -m pytest solutions/fuzzer/tests
jpamb interpret fuzzer-interpreter
```

## Running on the real JVM

`solutions/bytecode/InstrumentationAgent.java` is a Java agent that instruments the case classes and records the sites and int values a run reaches. It uses `java.lang.classfile`, so it needs JDK 24 or newer. Build it into `build/jvm/` from the root of the repository:

```bash
mkdir -p build/jvm/classes
javac -d build/jvm/classes $(find cases solutions/bytecode -name '*.java')
printf 'Premain-Class: solutions.bytecode.InstrumentationAgent\n' > build/jvm/manifest.txt
(cd build/jvm/classes && jar cfm ../agent.jar ../manifest.txt solutions/bytecode/InstrumentationAgent*.class)
```

`jvm_agent.py` runs one input on the JVM with the agent and reads the values the agent writes to a file when the JVM exits. `fuzzer-jvm-check` in `solutions/static` uses it.

## Notes

The interpreter gets 213 of 266 on JPAMB 0.10. The 53 cases it fails are in `Randomness` with `Smith` inputs, and in `Strings`, `CWE470` and `CWE502` with string methods like `startsWith`, `invokedynamic` and casts. We do not support those.

- We record ranges at comparisons, `idiv`, `irem` and array indices.
- The split points are 0 and the constants the method compares with. We get these from Adriana's syntactic analysis. `--split three` is the default and gives < c, = c and > c. `--split boundary` also gives c-1 and c+1 their own range.
- `instruction` only counts new instructions as new coverage and the other two also count new ranges.
- Full coverage means the instructions of the fuzzed method, of the case methods it calls and of the case methods those call.
- Arrays and strings get at most 8 elements, and a run stops with `*` after 1,000 steps.
- Without a bound the `range` version only stops on the run rules. `fuzzer-with-bound` from `solutions/static` takes the same options as `fuzzer` and gives every run the bound from the interval analysis. `range` then also stops when it has covered every range in the bound, and every line gets the analysis time, the size of the bound, how much of it was not covered and the covered ranges outside it.
- A method without parameters only gets one run, because every run would be the same.
- `fuzzer-extension` from `solutions/static` tries three ways to reach the ranges in the bound that `range` did not cover: going backwards along a run of a kept input, the binary search of Group 5 and fuzzing with ten times as many runs without new coverage. The first two test at most 20 values for each int parameter. It only works on methods with int parameters, and it writes one JSON line per range with the runs each way needed to reach it, or null.
- `fuzzer-results` from `solutions/static` prints the numbers the proposal asks for from a file of `fuzzer-with-bound`, and from a file of `fuzzer-extension` if it gets one.
- Negative array sizes give `negative array size` and arrays over 1,000,000 elements give `out of memory`. JPAMB does not list these outcomes.

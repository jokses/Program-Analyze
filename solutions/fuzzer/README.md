# Fuzzer

The fuzzer, the interpreter it runs the methods in, and the runner for the JVM. The [root README](../../README.md) explains the idea, the three versions and the terms used here (range, split point, split, bound). This README is about using the package. Install as described there, and run everything from the root of the repository.

## Running

Fuzz one method with all three versions and 10 seeds, and print a summary:

```bash
fuzzer 'jpamb.cases.Arrays.arraySumIsLarge:([I)V' --out arraySumIsLarge.jsonl
fuzzer-checks arraySumIsLarge.jsonl
```

Fuzz every method in `target/stats/cases.txt` with one seed. This took about 2.5 minutes on our machine:

```bash
fuzzer --all --seeds 1 --out all.jsonl
fuzzer-checks all.jsonl
```

`--split boundary` picks the other split, `--versions range` runs only the `range` version, and `fuzzer --help` lists all the options.

`fuzzer` writes one JSON line for each method, version and seed: the number of runs, the CPU time, why the version stopped, the outcomes, the instructions and ranges it covered, and the inputs it kept.

`fuzzer-checks` prints one line for each method, split, domain and version, over all its seeds. `fuzzer` uses no domain, which is shown as `-`. A line has:

- the number of seeds, the median number of runs and the median CPU seconds. Under `fuzzer-with-bound` the seconds of the `range` version include the analysis.
- how many seeds stopped for each reason.
- the outcomes JPAMB lists for the method that a seed did not find, and in how many seeds. "(with *)" means the seed also took more than 1,000 steps in a run, which gives the outcome `*`, so it may have found the outcome with more steps.
- with a bound, the median number of covered ranges out of the bound, and how many covered ranges lie outside the bound. That last number should always be 0.

Before these lines it lists the methods the interpreter crashes on, with the error.

### Building the agent

`solutions/bytecode/InstrumentationAgent.java` is a Java agent that instruments the case classes and records the int values a run sees at each comparison and division. It uses `java.lang.classfile`, so it needs JDK 24 or newer. Build it into `build/jvm/` from the root of the repository:

```bash
mkdir -p build/jvm/classes
javac -d build/jvm/classes $(find cases solutions/bytecode -name '*.java')
printf 'Premain-Class: solutions.bytecode.InstrumentationAgent\n' > build/jvm/manifest.txt
(cd build/jvm/classes && jar cfm ../agent.jar ../manifest.txt solutions/bytecode/InstrumentationAgent*.class)
```

## Files

- `fuzzer.py` is `fuzzer`. It runs the three versions on a method and decides when each stops. The split points come from the syntactic analysis in `solutions/syntactic`.
- `dynamic.py` is the interpreter, which also records the ranges. JPAMB runs it as `fuzzer-interpreter` and `fuzzer-analysis`. We followed the course notes on [selecting input](https://courses.compute.dtu.dk/02242/topics/dynamic-analysis.html#sec:selecting-inputs): `fuzz_input` makes random inputs, the constants in the method are used as a dictionary, and `Corpus` keeps the inputs that give new coverage. `Draw` and `fuzz_type` are from JPAMB's [extra-fuzzer commit](https://github.com/kalhauge/jpamb/commit/7a7a36af695aae8638be4983e291d456d4eff277).
- `checks.py` is `fuzzer-checks`, and it reads the JSON lines for the other commands too.
- `jvm_agent.py` runs one input on the JVM with the agent, and reads the values the agent writes to a file when the JVM exits. `fuzzer-jvm-check` in [`solutions/static`](../static/README.md) uses it.

## Notes

- Full instruction coverage means the instructions of the method and of every case method it reaches through calls.
- Without a bound, the `range` version stops only on the run limits. `fuzzer-with-bound` in [`solutions/static`](../static/README.md) gives it the bound.
- A method without parameters gets one run, because every run would be the same.
- Arrays and strings get at most 8 elements, and a run that takes more than 1,000 steps has the outcome `*`.
- A negative array size gives `negative array size`, and an array of more than 1,000,000 elements gives `out of memory`. JPAMB lists neither outcome.
- The interpreter crashes on the cases in `Randomness` with `Smith` inputs, and on those in `Strings`, `CWE470` and `CWE502`, which use string methods like `startsWith`, `invokedynamic` or arrays of objects. `fuzzer-results` leaves those methods out.

## Tests

Run the tests, and `dynamic.py` on JPAMB as interpreter and as analysis:

```bash
python -m pytest solutions/fuzzer/tests
jpamb interpret fuzzer-interpreter
jpamb analyse fuzzer-analysis
```

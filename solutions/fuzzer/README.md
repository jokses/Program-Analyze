# Fuzzer

We followed the course notes on [selecting input](https://courses.compute.dtu.dk/02242/topics/dynamic-analysis.html#sec:selecting-inputs) with `fuzz_input` for random inputs, the constants of the method as a dictionary and `Corpus` for keeping the inputs that give new coverage. `Draw` and `fuzz_type` are from Christian's [extra-fuzzer commit](https://github.com/kalhauge/jpamb/commit/7a7a36af695aae8638be4983e291d456d4eff277). Joachim's interpreter needed a few fixes for this (the pc of calls and errors, Java int arithmetic and bad array sizes).

The fuzzer has three versions. They all start with the empty input and one input per constant in the method. After that `random` makes random inputs, `instruction` mutates the inputs that ran a new instruction and `range` mutates the inputs that ran a new instruction or reached a new range. A version stops after 1,000 runs without new coverage or 100,000 runs in total, and `instruction` also stops when every instruction in the method has run.

## Running

From the root of the repository with the venv from `docs/python.md`:

```bash
uv pip install --editable solutions/fuzzer
fuzzer 'jpamb.cases.Dependent.divisionLoop:(I)V' --seeds 10 --out divisionLoop.jsonl
fuzzer --all --seeds 1 --split boundary --out all.jsonl
fuzzer-checks divisionLoop.jsonl all.jsonl
python -m pytest solutions/fuzzer/tests
```

`fuzzer` writes a JSON line per method, version and seed. `fuzzer-checks` prints the results for `divisionLoop` and `arraySumIsLarge` and the runs that crashed or hit 100,000 runs.

## Notes

- We record ranges at comparisons, `idiv`, `irem` and array indices.
- We split the ranges at 0 and at the int constants in the method. `--split three` is the default and gives < c, = c and > c, and `--split boundary` also gives c-1 and c+1 a range each.
- For `instruction` only a new instruction counts as new coverage. For the other two a new range counts too.
- Full instruction coverage only counts the instructions of the fuzzed method and not of the methods it calls.
- Arrays and strings have at most 8 elements, and a run gives `*` after 1,000 steps.
- `bound(method)` returns None until we have the interval analysis.
- A negative array size gives `negative array size` and more than 1,000,000 elements gives `out of memory`. JPAMB does not have these two outcomes.

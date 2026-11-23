# Fuzzer

We followed the course notes on [selecting input](https://courses.compute.dtu.dk/02242/topics/dynamic-analysis.html#sec:selecting-inputs). `fuzz_input` makes random inputs, the constants in the method are used as a dictionary and `Corpus` keeps the inputs that give new coverage. `Draw` and `fuzz_type` are from Christian's [extra-fuzzer commit](https://github.com/kalhauge/jpamb/commit/7a7a36af695aae8638be4983e291d456d4eff277).

The fuzzer has three versions, `random`, `instruction` and `range`. All of them first try the empty input and the constants in the method. `random` then makes random inputs. `instruction` mutates the inputs that ran a new instruction and `range` the inputs that ran a new instruction or hit a new range. They stop after 1,000 runs with nothing new or 100,000 runs in total. `instruction` also stops when it has run every instruction.

## Running

From the root of the repository with the venv from `docs/python.md`:

```bash
uv pip install --editable solutions/fuzzer
fuzzer 'jpamb.cases.Dependent.divisionLoop:(I)V' --seeds 10 --out divisionLoop.jsonl
fuzzer --all --seeds 1 --split boundary --out all.jsonl
fuzzer-checks divisionLoop.jsonl all.jsonl
python -m pytest solutions/fuzzer/tests
```

`fuzzer` writes one JSON line per method, version and seed. `fuzzer-checks` prints what we look at for `divisionLoop` and `arraySumIsLarge` and the runs that crashed or hit 100,000 runs.

## Notes

- We record ranges at comparisons, `idiv`, `irem` and array indices.
- The split points are 0 and the constants the method compares with. We get these from Adriana's syntactic analysis. `--split three` is the default and gives < c, = c and > c. `--split boundary` also gives c-1 and c+1 their own range.
- `instruction` only counts new instructions as new coverage and the other two also count new ranges.
- Full coverage means the instructions of the fuzzed method, of the case methods it calls and of the case methods those call.
- Arrays and strings get at most 8 elements, and a run stops with `*` after 1,000 steps.
- `bound(method)` returns None until we have the interval analysis.
- A method without parameters only gets one run, because every run would be the same.
- Negative array sizes give `negative array size` and arrays over 1,000,000 elements give `out of memory`. JPAMB does not have these outcomes though so we should be fine.

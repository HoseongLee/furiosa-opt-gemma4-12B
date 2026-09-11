# Changes

## 2026-09-18

### Fixture inputs no longer collapse to a fixed configuration

`sliding_attention_output`'s `x` (the simulated attention-head output feeding `o_proj`) was
drawn from `signs()`, so it was always exactly `bf16(+1)` or `bf16(-1)` -- never any other
value. That let a kernel's projection matmul get away with a sign-magnitude shortcut that
is exact for `+/-1` operands but wrong in general, since the test never exercised anything
else. `x` is now drawn from a continuous `[-1.0, 1.0)` range (`bf16_uniform`, not `signs`).

The same fixed-configuration problem existed for two other values that never varied between
test runs:

- **RoPE/cache-offset position** was hardcoded at `137`. Now drawn per run from the shared
  fixture PRNG (never `0`, since an all-zero position makes RoPE the identity rotation for
  every frequency -- itself a fixed configuration a kernel could special-case).
- **NVFP4 global scales** (`up`/`gate`/`down`) were pinned to the real checkpoint's numbers
  (`9600.0`, `9600.0`, `12928.0`). Now drawn per run from a `[2048.0, 16384.0)` range of the
  same order of magnitude, so a kernel can't special-case the checkpoint's literal values.

### Each kernel is graded over multiple independently-seeded runs, not one

Previously each kernel was measured as a single invocation against one fixed fixture.
`generate_references.py` now bakes `RUNS` (3) independent draws of every input into one
fixture, keyed `run{n}.{test}.{label}`, and `test_kernels.rs` sweeps all of them -- batched
per kernel -- from a single binary invocation (one local run, one remote job) rather than
replaying the same fixed numbers every time. A kernel must pass every run to count as
correct; the score reported per kernel is the *median* cycle count across the sweep, not
any single run's number, so one unusually fast or slow run doesn't move the number that's
graded. 


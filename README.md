# Gemma-4 Kernel Optimization Competition on Furiosa RNGD

This repository is the competition skeleton code for optimizing Gemma-4-12B-it on Furiosa
RNGD. Stage 1 focuses on three decoder-layer kernels; Stage 2 will focus on end-to-end
(E2E) performance. Competitors modify the allowed implementation while preserving the
public interface and the model's numerical behavior.

The two-stage format is confirmed. Only the specific submission and scoring values marked
**TBD** remain to be announced.

## What you are optimizing

The three Stage 1 kernels are declared in `src/ops.rs`:

| Kernel | Operation |
|---|---|
| `ops::sliding_project_qkv` | RMSNorm, Q/K/V projection, Q/K RMSNorm, RoPE, and K/V ring-cache writes |
| `ops::sliding_attention_output` | Head broadcast, O projection, post-attention RMSNorm, and residual add |
| `ops::decoder_feedforward` | RMSNorm, GeGLU MLP, post-FF RMSNorm, residual add, and layer gate |

The kernels use the model's existing quantized weights and tensor layouts. Each kernel is
measured across 3 independently-seeded input configurations for the Stage 1 test (see
"Grading criteria" below), not a single fixed invocation.

Stage 2 covers the complete E2E model-serving path except for the public API endpoint in
`src/api/`. This includes model execution, host orchestration, tokenization, image and
audio preprocessing, and runtime integration.

## Competition format

The competition has two stages. The scope of each stage is confirmed below; the remaining
unknown submission and scoring values are listed explicitly.

### Stage 1 — Kernel optimization

Stage 1 is for optimizing the three kernels listed above. A dedicated grading server will
run the provided `src/bin/test_kernels.rs` against each submission. This test is the source
of truth for Stage 1 correctness and kernel performance.

The Stage 1 test checks:

1. **Buildability:** the allowed code compiles with the competition toolchain.
2. **Correctness:** each kernel satisfies the published tolerances on every one of 3
   independently-seeded input configurations, not just one -- a kernel that special-cases
   a fixed input fails as soon as a different configuration exposes it.
3. **Performance:** the test reports each kernel's median RNGD cycle count across those 3
   runs, so one unusually fast or slow run doesn't move the number that's graded.

Stage 1 values to be finalized:

- **TBD:** submission deadline.

### Stage 2 — End-to-end optimization

Stage 2 is for optimizing the complete Gemma-4-12B-it E2E path, excluding the public API
endpoint in `src/api/`.

Stage 2 values to be finalized:

- **TBD:** benchmark input count, prompt lengths, and output lengths;
- **TBD:** number of benchmark repetitions and RNGD configuration;
- **TBD:** correctness thresholds and E2E performance metric;
- **TBD:** submission deadline, score formula, and tie-break rule.

## Grading criteria

The competition uses one grading policy across both stages. Correctness is a hard gate: a
submission that fails a required correctness check receives no performance credit, even if
it is faster.

For the Stage 1 kernel evaluation, `src/bin/test_kernels.rs` defines the following tolerances:

| Kernel | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| `sliding_project_qkv` | `0.04` | `1e-2` |
| `sliding_attention_output` | `0.05` | `1e-2` |
| `decoder_feedforward` | `0.01` | `1e-2` |

The grading server will measure performance using the official evaluation. The Stage 1 test
reports each kernel's median RNGD cycle count across 3 independently-seeded runs; the
Stage 2 E2E performance metric is TBD.

The following scoring values are still TBD:

| Field | Planned rule |
|---|---|
| Performance metric and weighting | **TBD** |
| Failed or timed-out run | **TBD** |
| Reproducibility and code review | **TBD** |

Schedule makespan is a useful development metric, but it is not a substitute for official
grading results.

## Stage 1 rules: skeleton contract

The following rules apply to Stage 1 kernel submissions. Stage 2 follows the E2E scope
described above; its remaining submission rules are listed in the Stage 2 section.

The Stage 1 skeleton is intentionally fixed so that submissions remain comparable.

1. **Keep device function names and signatures unchanged.** The name, parameters, types,
   and return type of every `#[device]` function are part of the evaluator contract.
2. **Only permitted implementation changes are graded.** Changes to `src/device/` and the
   function bodies in `src/ops.rs` are included in the evaluation. Everything else is
   ignored, including `src/ops_vision.rs`, `src/ops_audio.rs`, `src/axes.rs`, `src/host/`,
   `src/api/`, `src/bin/`, and `src/lib.rs`.
3. **Keep kernel module paths stable.** `src/ops.rs`, `src/ops_vision.rs`, and
   `src/ops_audio.rs` must remain at the crate root because compiled kernel names include
   `module_path!()`.

Shared code may affect more than one path. In particular, `device/shared/rmsnorm.rs` and
`device/shared/mlp.rs` are also used by full attention, vision, and audio code. Check the
impact of a shared change before measuring a result.

## Evaluation commands

### Prepare the public fixture

Run this once when `ref/fixtures.safetensors` is missing:

```sh
python3 scripts/generate_references.py
```

The fixture contains expected outputs and checksums. Inputs are generated deterministically
from the shared PRNG on both the Python and Rust sides.

### Optimize a kernel

Follow [OPTIMIZATION.md](OPTIMIZATION.md) for the schedule-dump, bottleneck-analysis,
kernel-tuning, and makespan-comparison workflow.

### Run the public RNGD check

```sh
./scripts/rngd_test.sh
```

This builds the test binary, submits it through the Arena server, and reports accuracy
and real RNGD cycle counts. Use `--no-build` to reuse the latest binary or `--no-wait` to
submit without waiting for the result.

For a locally available RNGD setup, the repository also provides:

```sh
./scripts/local_test.sh
```

### Submit for grading

```sh
cargo binstall moa-submitter-cli

moa-submitter login
moa-submitter submit         # run from the repository root
moa-submitter status         # the state of every submission you have made
moa-submitter status <id>    # the cycle counts and score of one submission
moa-submitter log <id>       # the complete log, stage by stage
```

`submit` uploads `src/ops.rs` and everything under `src/device/`, so keep the original
directory structure. Run it from the repository root, or pass
`--source path/to/furiosa-opt-gemma4-12B`.

Each submission is scored as the geometric mean of its speedup over the baseline across the
three kernels. Only a team's highest score appears on the leaderboard.

## Toolchain

The supported development environment is x86_64 Ubuntu 22.04 or newer with GLIBC 2.34 or
newer:

```sh
sudo apt install build-essential libclang-dev
sudo apt install gcc-aarch64-linux-gnu

rustup toolchain install nightly-2026-05-01
cargo +nightly-2026-05-01 install cargo-binstall
cargo +nightly-2026-05-01 binstall cargo-furiosa-opt@0.8.1
cargo install furiosa-schedule-viewer
```

Configure the Furiosa Arena CLI once before using `scripts/rngd_test.sh`:

```sh
cargo binstall furiosa-arena-cli

furiosa-arena login
```

The cli commands used for troubleshooting are:

| Command | Purpose |
|---|---|
| `furiosa-arena submit <file>` | Submit a script or binary |
| `furiosa-arena status <id>` | Check job state |
| `furiosa-arena logs <id> --follow` | Stream job output |
| `furiosa-arena list` | List your jobs |
| `furiosa-arena cancel <id>` | Cancel a queued or running job |

## References

- [Programming Tensor Contraction Processors](https://developer.furiosa.ai/furiosa-opt/book)
  — mappings, movement, computation, scheduling, and tuning.
- [`furiosa_opt_std` API docs](https://docs.rs/furiosa-opt-std/latest/furiosa_opt_std/)
  — tensor types, mapping expressions, and engine modules.
- [furiosa-arena-cli docs](https://github.com/kreatinj/furiosa-arena-cli#installation)
- [moa-submitter-cli](https://github.com/micro2026-moa/moa-submitter-cli) — the submission CLI.
- [OPTIMIZATION.md](OPTIMIZATION.md) — the Stage 1 kernel optimization workflow.
- [ARCHITECTURE.md](ARCHITECTURE.md) — repository layout and host/RNGD split.
- [SERVING.md](SERVING.md) — running the model as an HTTP server.

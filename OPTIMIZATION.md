# Stage 1 Kernel Optimization Guide

This guide contains the original kernel-optimization workflow for Stage 1 of the
competition. It explains how to inspect a compiled schedule, find a bottleneck, change a
kernel, and compare results. Stage 1 grading is performed with
[`src/bin/test_kernels.rs`](src/bin/test_kernels.rs); the schedule is a development aid, not
the grading authority.

## 1. Dump a schedule

```sh
mkdir -p target/schedules
cargo furiosa-opt compile ops::sliding_project_qkv --exact \
    --dump-schedule target/schedules/sliding_project_qkv.json
```

The `--dump-*` flags are single-kernel options: one invocation writes one file, so dump
the three kernels one at a time.

`--exact` is important because the positional filter is a substring match by default.
Some kernel names are prefixes of others. For example,
`ops::sliding_attention` also matches `ops::sliding_attention_output` unless the filter is
exact.

The other dump flags are described in
[Kernel Optimizer](https://developer.furiosa.ai/furiosa-opt/book/tools/kernel-optimizer.html):
`--dump-visa`, `--dump-ir`, `--dump-dfg`, `--dump-graph`, and `--dump-summary` (which takes
a directory rather than a file).

## 2. Open the schedule viewer

```sh
furiosa-schedule-viewer
```

The viewer binds to `127.0.0.1:9254` and opens in a browser by default. Drag the schedule
JSON onto the drop zone, or click the drop zone to select it. `--host` and `--port` change
the bind address.

Use the viewer to inspect:

- **Nodes:** name, lifetime, context, connected nodes, and source location;
- **Cycle range:** a time window to isolate a suspicious region;
- **Brush:** a memory-address range instead of a time range.

The viewer shows the static execution plan. It does not show values or official RNGD
performance, and its instructions do not always correspond one-to-one with source-level
operators.

Reference: [Schedule Viewer](https://developer.furiosa.ai/furiosa-opt/book/tools/schedule-viewer.html).

## 3. Find the bottleneck

Start with the schedule's overall span, then trace the longest-lived nodes to their inputs,
outputs, contexts, and source lines. Gaps usually indicate dependency waits. A context that
stays busy while others are idle is often the limiter for that interval.

The main contexts are:

- **`DmaEngine`:** movement of weights and activations is limiting;
- **`SubContext`:** register-file staging and preloads are limiting;
- **`MainContext` / `VectorEngine`:** vector computation such as softmax, RMSNorm, or
  casts is limiting.

`MainContext` and `SubContext` contend for the Tensor Unit pipeline. Overlap is therefore
bounded: total time can approach the sum of their work, while ideal overlap approaches the
larger contributor.

On DMA nodes, inspect `util` in addition to duration. Low utilization can indicate an
awkward access pattern rather than an unavoidable bandwidth limit. Partition-crossing,
strided, and non-contiguous accesses are common causes. Alignment and HBM bank conflicts
can also be expensive; see the
[Memory Performance](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/memory-performance.html)
and [Schedule](https://developer.furiosa.ai/furiosa-opt/book/scheduling/schedule.html)
chapters for details.

The scheduler splits the TRF into halves automatically. A tensor that fits in half the file
can share the other half with another operation, and both halves share banks.

### Example: `ops::sliding_project_qkv`

```sh
cargo furiosa-opt compile ops::sliding_project_qkv --exact \
    --dump-schedule target/schedules/sliding_project_qkv.json
```

In one earlier revision, the schedule spanned roughly 116,600 cycles. `MainContext` was
busy for about 96% of the span, making compute the larger lever. The longest node was a
`MainContext` fetch-and-switch of roughly 62,000 cycles at:

```
--> src/device/layout.rs:16
```

That source location was a broadcast helper, not the projection itself. Three projection
weight loads were also visible on `DmaEngine`, but they overlapped with the longer compute
interval. Compare contributors before optimizing an individual node.

These figures are historical examples and will change as the skeleton changes. The
diagnostic method is the reusable part.

### Inspect the raw schedule JSON

The schedule JSON is plain and scriptable. Each entry in `instructions` includes `tpe`,
`contexts`, `lifetime` (`begin`/`end`), `util` (with `total_util`), and a `description`
containing the source location and expression. Each entry in `tensors` includes
`buffer_type`, `address`, `size`, and `shape`. Makespan is
`max(instruction.lifetime.end)`.

Further reading: [Diagnosis](https://developer.furiosa.ai/furiosa-opt/book/scheduling/diagnosis.html).

## 4. Change the kernel

Choose the optimization lever that matches the bottleneck:

1. **Execution-engine path:** use this when the current resource is the bottleneck.
2. **Tile or split shape:** use this for avoidable serial work or an ill-fitting reduction.
3. **Mapping, padding, or transfer boundary:** use this when movement or an address
   dependency limits the interval.

The relevant API is documented in the
[`prelude`](https://docs.rs/furiosa-opt-std/latest/furiosa_opt_std/prelude/index.html),
including [`m!`](https://docs.rs/furiosa-opt-std/latest/furiosa_opt_std/prelude/macro.m.html),
[`Tensor`](https://docs.rs/furiosa-opt-std/latest/furiosa_opt_std/prelude/struct.Tensor.html),
the memory-tier aliases, and the
[`contraction`](https://docs.rs/furiosa-opt-std/latest/furiosa_opt_std/prelude/contraction/index.html)
and [`vector`](https://docs.rs/furiosa-opt-std/latest/furiosa_opt_std/prelude/vector/index.html)
modules.

## 5. Compare the result

Re-dump the kernel and compare its makespan with the previous schedule. Keep the old JSON
files so that improvements remain reproducible. Also record which context dominates after
the change; a shift from `MainContext` to `DmaEngine`, for example, indicates a different
next optimization target.

Makespan is static: it describes the compiler's plan. Confirm any promising change with
the Stage 1 test on RNGD:

```sh
./scripts/rngd_test.sh
```

Do not report a cycle improvement without a reproducible schedule comparison or separately
documented RNGD evidence. A faster but numerically incorrect kernel does not receive Stage 1
performance credit because accuracy is a hard grading gate.

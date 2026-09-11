# Architecture

How this crate is laid out and what every file does. If you are here to make the
kernels faster, start with [README.md](README.md); this document is the map you
consult when you need to know where something lives. For the HTTP server and
thinking mode, see [SERVING.md](SERVING.md).

## How the project is organised

The single most important thing to understand is the **device / host split**, and the
narrow seam between them:

```
                      ops.rs · ops_vision.rs · ops_audio.rs
   host/  ─────────────────►  (the #[device] entry points)  ─────────────►  device/
   CPU code: loops over          one dispatch = one token                   the kernel
   layers, positions,            position, image patch,                     building
   requests                      or audio frame                             blocks
```

- **`device/`** is everything that compiles to kernels for Furiosa RNGD. It never calls
  into `host/`.
- **`host/`** is everything that runs on the CPU. It never contains kernel code.
- **`ops*.rs`** are the only meeting point: the `#[device]` functions the host launches.

Two consequences of the hardware shape drive the whole layout:

1. **A kernel processes exactly one unit of work.** There is no batch or sequence
   dimension inside a kernel. Every loop — over the 48 layers, over token positions,
   over image patches, over audio frames — lives on the host.
2. **A kernel's compiled artifact is named `module_path!()::fn_name`.** That is why
   `ops`, `ops_vision` and `ops_audio` are declared at the crate root and stay there:
   moving them renames every kernel and breaks the tooling that refers to them by name.

The model also has **two distinct attention geometries**, which is why `device/` is split
by geometry rather than by operation. The geometries are intentionally kept separate.

## Directory tree

```
src/
├── lib.rs                      crate root: module list, Chip, model constants
├── axes.rs                     every named tensor dimension, annotated
│
├── ops.rs                      text #[device] entry points
├── ops_vision.rs               vision #[device] entry points
├── ops_audio.rs                audio #[device] entry point
│
├── device/                     kernel building blocks (never used from host/)
│   ├── mod.rs
│   ├── layout.rs               Cluster/Slice/Replicated + broadcast helpers
│   ├── sliding/                the 40 sliding-window layers
│   │   ├── mod.rs
│   │   ├── projection.rs
│   │   ├── rmsnorm.rs
│   │   ├── rope.rs
│   │   └── attention.rs
│   ├── full/                   the 8 full-attention layers
│   │   ├── mod.rs
│   │   ├── projection.rs
│   │   ├── rmsnorm.rs
│   │   ├── rope.rs
│   │   └── attention.rs
│   ├── shared/                 used by both geometries
│   │   ├── mod.rs
│   │   ├── rmsnorm.rs
│   │   ├── residual.rs
│   │   ├── mlp.rs
│   │   └── lm_head.rs
│   ├── vision/                 the vision encoder
│   │   ├── mod.rs
│   │   ├── projection.rs
│   │   └── layernorm.rs
│   └── audio/                  the encoder-free audio embedder
│       ├── mod.rs
│       └── projection.rs
│
├── host/                       CPU-side orchestration and preprocessing
│   ├── mod.rs
│   ├── runtime.rs              persistent device state + the layer loop
│   ├── load.rs                 checkpoint -> HBM
│   ├── generate.rs             prefill-then-decode loop
│   ├── tokenizer.rs            chat templating and BPE
│   ├── sampling.rs             temperature / top-k / top-p
│   ├── image.rs                image decode, resize, patchify
│   └── audio.rs                WAV decode, resample, framing
│
├── api/                        OpenAI-compatible HTTP server
│   ├── mod.rs
│   ├── schema.rs               request/response types
│   ├── worker.rs               the dedicated generation thread
│   ├── server.rs               hand-rolled HTTP/1.1 layer
│   └── handlers.rs             per-route glue
│
└── bin/
    ├── gemma4.rs               CLI entry point (one prompt, prints the answer)
    ├── server.rs               API server entry point
    └── test_kernels.rs         kernel fixture test for the 3 decoder-layer kernels;
                                 accuracy + RNGD cycle counts (see its module doc)

scripts/
├── generate_references.py      write ref/fixtures.safetensors (expected outputs only)
├── fixture_prng.py             input synthesis; mirrored by src/bin/test_kernels.rs's prng module
├── local_test.sh               run the kernel tests against local Furiosa RNGD
├── rngd_test.sh                run them on remote Furiosa RNGD via the Arena server
│
├── run.sh                      CLI smoke test via src/bin/gemma4.rs
├── run_server.sh               launch the API server with local Furiosa RNGD
│
├── rngd/                       not run by hand; shipped to the rngd worker
│   └── remote_entrypoint.sh    the submitted job's entrypoint (runs on the worker)
│
└── reference/                  not run; use as the architecture reference
    └── gemma4.py               standalone pure-PyTorch implementation
```

## What each file does

### Crate root

| File | Purpose |
|---|---|
| `lib.rs` | Crate documentation, the module list, the `Chip` type, and the four model constants (`LAYERS`, `EMBED_SCALE`, `LOGIT_SOFTCAP`, `EPS`). Deliberately thin — it declares the structure rather than containing logic. |
| `axes.rs` | **Read this first.** Every named tensor dimension in the crate and the single source of truth for shapes. Check here whenever a shape mismatch appears. Currently unannotated — the `axes!` macro parses only `Name = literal`, so any explanatory text has to live as `//` comments, not `///`. |

### Device entry points

These are the functions the host launches. Each is one RNGD dispatch.

| File | Purpose |
|---|---|
| `ops.rs` | The 10 text kernels. Per decoder layer the host calls `{sliding,full}_project_qkv` → `sliding_attention` (or `full_attention_first_page`/`full_attention_page`, one dispatch per KV-cache page) → `{sliding,full}_attention_output` → `decoder_feedforward`, then `final_norm_and_logits` once at the end. Also holds `embed_token`. |
| `ops_vision.rs` | The 4 vision kernels: `patch_embed` → `add_position_and_norm` → `project_to_text_embedding` turn one image patch into a soft token. `layernorm` is exposed separately so it can be tested in isolation. |
| `ops_audio.rs` | The single audio kernel, `audio_project_frame`: RMS-normalizes one 640-sample waveform frame and projects it into text space. |

### `device/` — kernel building blocks

| File | Purpose |
|---|---|
| `layout.rs` | How work spreads across the hardware: the `Cluster`, `Slice` and `Replicated` types, plus the `broadcast_*` helpers that copy one value to all 256 slices before a contraction. |
| `sliding/projection.rs` | Q/K/V/O projections for a sliding-window layer. Weight-only 8-bit, not W8A8: `f8e4m3` weights with one bf16 scale per output channel, widened to bf16 and contracted against an unquantized bf16 activation, with the scale applied after the contraction. |
| `sliding/rmsnorm.rs` | Per-head Q/K/V RMSNorm. Q and K use learned weights; V has none. |
| `sliding/rope.rs` | Rotary embedding over the full head dim, θ=10,000. |
| `sliding/attention.rs` | Masked softmax attention over the whole ring KV cache in a single pass. |
| `full/projection.rs` | Q/K/O projections for a full-attention layer. There is **no V projection** — the checkpoint has no `v_proj` for these layers. |
| `full/rmsnorm.rs` | Per-head Q/K/V RMSNorm for one KV head. `normalize_value` derives V from the *raw* K projection, which is what the real model does. |
| `full/rope.rs` | Rotary embedding at θ=1,000,000, where only the first quarter of angles rotate. |
| `full/attention.rs` | Flash-attention-style online softmax across KV cache pages, one dispatch per page, with the running max/sum/output carried through HBM. |
| `shared/rmsnorm.rs` | RMSNorm over the text hidden state — the input, post-attention, pre/post feed-forward and final norms. |
| `shared/residual.rs` | Residual-stream adds and the per-layer scalar gate. |
| `shared/mlp.rs` | The GeGLU feed-forward network. This is the model's NVFP4 component: 4-bit packed weights with a per-16-element scale *and* a per-matrix global scale. |
| `shared/lm_head.rs` | Hidden state to vocabulary logits. Defines its own cluster/slice split because the vocabulary is far wider than anything else. |
| `vision/projection.rs` | The two vision contractions: patch pixels → embedding, and encoder output → text hidden size. |
| `vision/layernorm.rs` | LayerNorm, at both the patch-pixel and patch-embedding widths. The **only** LayerNorm in the model; the text path has none. |
| `audio/projection.rs` | Unweighted RMSNorm followed by a plain BF16 contraction, turning one waveform frame into a text-space vector. |

### `host/` — CPU side

| File | Purpose |
|---|---|
| `runtime.rs` | The orchestration layer. `Workspace` owns everything that outlives a token — KV caches, RoPE tables, causal masks, scratch. `Decode` holds per-position scratch and runs the loop over all 48 layers. Also encodes vision patches and audio frames into soft tokens. |
| `load.rs` | Reads the safetensors checkpoint into HBM, with one struct per layer kind. Picks the layer kind by index, and handles the checkpoint's mixed precision: fp8 per-channel scales for attention, real NVFP4 for the MLP only. |
| `generate.rs` | The prefill-then-decode loop over token positions: splices image and audio placeholders into the prompt, samples each next token, handles stop strings safely for streaming, and splits the model's thinking channel out of its answer. |
| `tokenizer.rs` | Chat templating and BPE encode/decode, including the incremental decoder used for streaming and the `thinking` flag that reshapes the prompt for reasoning mode. |
| `sampling.rs` | Temperature, top-k and top-p. No device dependency. |
| `image.rs` | Decodes an image, resizes it to a whole number of merged patches, and cuts it into patches. Host-only. |
| `audio.rs` | Decodes a WAV (PCM 8/16/24/32-bit or float32, any channel count), downmixes to mono, resamples to 16 kHz, and cuts it into 640-sample frames with the tail zero-padded. Capped at 30 seconds. Host-only. |

### `api/` — the HTTP server

| File | Purpose |
|---|---|
| `mod.rs` | Just the module declarations (`schema`, `server`, `worker`, `handlers`) — no logic of its own. |
| `schema.rs` | Request and response types, close enough to OpenAI's wire format that stock SDK clients work. |
| `worker.rs` | The dedicated generation thread. This exists because `Device::new()` returns a non-`Send` guard, so the device cannot be moved between threads — one thread owns it, the model and the workspace for the process lifetime. **Read this before changing how the server is threaded.** |
| `server.rs` | A minimal synchronous HTTP/1.1 layer on `std::net`, one thread per connection, including the route table (`fn route`). Do not assume full HTTP/1.1. |
| `handlers.rs` | Per-route glue: validate a request, decode any image or audio, decide whether thinking mode is on (`wants_thinking`), turn it into a job, and shape the worker's events back into OpenAI responses. |

### `bin/`

| File | Purpose |
|---|---|
| `gemma4.rs` | The CLI entry point, driven by `scripts/run.sh`. Takes one prompt (plus optional `--image`/`--audio`) and prints the answer. |
| `server.rs` | The API server entry point. Reads `RNGD_MODEL_DIR`, `GEMMA4_API_ADDR` and `GEMMA4_API_KEY` from the environment. |
| `test_kernels.rs` | Kernel fixture test for the 3 decoder-layer kernels (`sliding_project_qkv`, `sliding_attention_output`, `decoder_feedforward`), driven by a fixture written by `scripts/generate_references.py`. |

### Scripts

| File | Purpose |
|---|---|
| `scripts/generate_references.py` | Writes `ref/fixtures.safetensors` by running `scripts/reference/gemma4.py`'s own checkpoint-matched reference modules — no `transformers` dependency. Stores **only expected outputs** (~120 KB); every input is synthesized on both sides instead. Needs no checkpoint. Run it by hand whenever a kernel's expected output changes — no test script generates it for you. |
| `scripts/fixture_prng.py` | The input synthesis, byte-for-byte mirrored by the `prng` module in `src/bin/test_kernels.rs`. Values come from a stateless counter hash, so each element depends only on its tensor's name and index and adding a test never moves another's bytes. |
| `scripts/local_test.sh` | Builds and runs the 3 kernel cases against local Furiosa RNGD, compares them to the fixture, and prints RNGD cycle counts alongside accuracy by default (`FURIOSA_OPT_PROFILE=info`; set `=off` to skip). Fails with instructions if the fixture is missing. |
| `scripts/rngd_test.sh` | Runs the same tests on remote Furiosa RNGD through the Arena server: builds `test_kernels`, submits it with `rngd/remote_entrypoint.sh` and the fixture, polls, and prints the log. Needs `$FURIOSA_ARENA_URL` and a prior `furiosa-arena login`. `--no-build` reuses the last build; `--no-wait` submits and returns immediately. |
| `scripts/rngd/remote_entrypoint.sh` | The rngd job's entrypoint, running on the worker — never invoked by hand; `rngd_test.sh` stages it into the submission. POSIX `sh`; defaults `FURIOSA_OPT_PROFILE=info` so the remote run collects cycles too, and copies the binary before `chmod`ing it because companion files arrive owned by another uid. |
| `scripts/run.sh` | CLI smoke test through `src/bin/gemma4.rs`. Arguments pass through, so `./scripts/run.sh --image path/to.png what is this` works. |
| `scripts/reference/gemma4.py` | A standalone pure-PyTorch implementation of the architecture (`torch` only — `transformers` is optional, for weight/config loading only). Exists to be read as ground truth beside `axes.rs`, and its `RMSNorm`/`MLP`/`rotate_half` are imported directly by `scripts/generate_references.py` as the reference implementation for the kernel tests. |
| `scripts/run_server.sh` | Launches the API server with local Furiosa RNGD. Start the server with this script rather than executing the built binary directly. `RNGD_MODEL_DIR` must point at the checkpoint or its parent; `GEMMA4_API_ADDR` (default `0.0.0.0:8000`) and `GEMMA4_API_KEY` (unset = no auth) configure the listener. |

## Request flow

A chat request with an image or audio attachment travels:

```
handlers.rs   decode the attachment      image.rs / audio.rs   ->  patches / frames
     │        render the chat template   tokenizer.rs
     ▼
worker.rs     queue one job, run it alone on the device-owning thread
     ▼
generate.rs   splice placeholders, then loop over positions
     │           image position ─► ops_vision::*  ─┐
     │           audio position ─► ops_audio::*   ─┼─► soft token replaces the embedding
     │           text position  ─► ops::embed_token┘
     ▼
runtime.rs    for each position, run 48 layers through ops::*, then the lm_head
     ▼
sampling.rs   pick the next token, feed it back, stream the text out
```

Requests are served strictly one at a time: one `Workspace` is one conversation's entire
KV cache, so there is no batching and no interleaving. A request under load waits its
turn in the queue.

# Serving

Running the model as an OpenAI-compatible HTTP server. This is separate from kernel
optimization — if you are here to make the kernels faster, see
[README.md](README.md). For where the server code lives, see
[ARCHITECTURE.md](ARCHITECTURE.md).

## Prerequisites

Access to Furiosa RNGD and a checkpoint is required. Unlike the kernel-optimization loop,
the server has no hardware-free path: it loads weights into HBM and runs the model.

- The toolchain from [README.md](README.md#toolchain).
- A Furiosa RNGD environment, local or through the Arena server.
- A Gemma-4-12B-it checkpoint in safetensors form.

## Starting the server

```sh
./scripts/run_server.sh
```

Start the server with this script rather than executing the built binary directly — the
script goes through `cargo furiosa-opt run`, which compiles the `#[device]` kernels before
launching. A plain `cargo run` builds for the CPU and will not work.

`src/bin/server.rs` reads three environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `RNGD_MODEL_DIR` | *required* | The checkpoint directory, or its parent |
| `GEMMA4_API_ADDR` | `0.0.0.0:8000` | Listen address |
| `GEMMA4_API_KEY` | unset | Bearer token; unset means no auth |

## Concurrency

Requests are served strictly one at a time. One `Workspace` is one conversation's
entire KV cache, so there is no batching and no interleaving — a request under load
waits its turn in the queue. This is a deliberate consequence of the device
being non-`Send`: a single thread owns the device, the model and the workspace for the
process lifetime. See `api/worker.rs` before changing how the server is threaded.

## CLI

For a quick check without the HTTP layer:

```sh
./scripts/run.sh what is the capital of France
./scripts/run.sh --image path/to.png what is this
```

## Thinking mode

`/v1/chat/completions` can let the model reason before it answers. Either spelling works —
`chat_template_kwargs` (the vLLM/SGLang convention) is checked first:

```jsonc
{ "chat_template_kwargs": { "enable_thinking": true } }   // or:
{ "reasoning_effort": "medium" }   // "none"/"minimal" means off; the template has no levels
```

With neither present thinking is off, matching the chat template's own default. The
reasoning comes back separated from the answer, under the field name the OpenAI-compatible
ecosystem settled on:

```jsonc
{ "choices": [ { "message": {
    "role": "assistant",
    "reasoning_content": "3 × 4 = 12, 3 × 5 = 15, 4 × 5 = 20 …",
    "content": "47\nJustification: …"
} } ] }
```

Streaming puts the same text in `delta.reasoning_content`; a chunk never carries both
fields. The field is omitted entirely when the model did not reason, so a client that knows
nothing about thinking sees an unchanged response shape.

**Budget for it.** Reasoning tokens are drawn from the same `max_tokens` allowance as the
answer, and the model is not brief: a question that answers fine in 200 tokens can easily
spend 500 reasoning first. Too small a budget returns `finish_reason: "length"` with a
populated `reasoning_content` and an **empty** `content`. Roughly 1000 tokens is a sane
floor.

The whole switch is a matter of prompt shape rather than a second decoding path — see
`api::handlers::wants_thinking` for how a request turns thinking on, and `host::generate`
for how the two channels are pulled apart.

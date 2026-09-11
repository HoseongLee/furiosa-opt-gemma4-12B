#!/usr/bin/env python3
"""Generate `ref/fixtures.safetensors`: the expected output of every decoder-layer kernel
under test.

`src/bin/test_kernels.rs` replays these on hardware and compares. This is the only
kernel-test path in the crate.

**The reference is `scripts/reference/gemma4.py`**, this repo's own checkpoint-matched pure-PyTorch
reimplementation of the architecture (not the upstream `transformers` package, which is
not a dependency of this script at all). Its `Gemma4RMSNorm`, `Gemma4MLP` and
`rotate_half` are called directly as the reference implementation. `axes.rs` and this
script's literal shape constants below independently pin down the same checkpoint shapes,
so the two are a cross-check on each other, but neither is validated against a second,
independently-authored implementation the way an upstream-`transformers`-backed reference
would be.

Each `gen_*` function drives the relevant piece of `reference/gemma4.py` for one kernel and records
its expected output. Only the parts under test are ever run; nothing here executes 48
layers.

**Inputs are not stored.** Every kernel input is synthesized from `fixture_prng`, which
`test_kernels.rs` reimplements byte-for-byte, so the fixture carries only expected outputs
and stays around 120 KB rather than the ~2.3 GB the inputs would need. A checksum of every
synthesized input travels with them so a divergence between the two implementations fails
by name instead of as a mysterious numeric error.

No checkpoint is required: the config values below are literals. The fixture bakes in
`RUNS` independent draws of every value -- including the RoPE position and the NVFP4
global scales -- keyed `run{n}.{test}.{label}`, so `test_kernels.rs` can sweep all of them
from one binary invocation instead of replaying the same numbers every time. This is the
same mechanism `sliding_attention_output` used to fail: a kernel can special-case whatever
the fixture always hands it, so nothing that reaches a kernel as data should be a single
fixed value across every run.

Adding a test: write a `gen_*` function that takes a `run: int` and returns
`(outputs, checksums)`, add it to `TESTS`, and add the matching shim and tolerance row in
`src/bin/test_kernels.rs`. The two registries are keyed by the same name and are kept in
step by hand.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
import fixture_prng as prng
import gemma4

CRATE = Path(__file__).resolve().parent.parent
FIXTURE = CRATE / "ref" / "fixtures.safetensors"

RUNS = 3

H, L, W = 3840, 15360, 262144
NS, GS, DS, QS, PS, TS = 8, 2, 256, 4096, 2048, 1024
GF, DF, QF, PF, TF = 16, 512, 8192, 512, 512

EPS = 1e-6

GLOBAL_SCALE = (2048.0, 16384.0)
"""Span for the three NVFP4 global scales. Order-of-magnitude matches the real checkpoint
(~9600/~9600/~12928) but the actual values are drawn per run rather than pinned to those
checkpoint numbers, so a submission can't special-case the real constants."""

F4_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

WEIGHT_EXP = (7, 14)
LOCAL_SCALE_EXP = (8, 10)
ROW_SCALE = (0.0, 1e-3)
UNIT = (0.0, 1.0)
RMS_WEIGHT = (0.75, 1.25)
ACTIVATION = (-1.0, 1.0)


def _run_pos(run: int) -> int:
    """The RoPE/cache-offset position for run `run`, drawn from the shared PRNG.

    `test_kernels.rs` derives the identical value for the same `run`, and every downstream
    tensor that depends on it (`cos`, `sin`, `kv_offset`, `rope_offset`) is checksummed, so
    the two sides failing to agree on `pos` surfaces immediately as a checksum mismatch
    rather than a silently wrong position. Never 0: an all-zero position makes RoPE the
    identity rotation for every frequency, which is exactly the kind of single fixed
    configuration a kernel could special-case.
    """
    word = prng.words(f"run{run}.global.pos", 1)[0]
    return 1 + int(word % np.uint64(2047))


LAYER_SCALAR = 0.375


def _tensor(storage: np.ndarray, dtype: torch.dtype, shape) -> torch.Tensor:
    """Reinterpret raw storage bytes as a torch tensor, without any float conversion."""
    return torch.frombuffer(bytearray(storage.tobytes()), dtype=dtype).reshape(shape)


class Synth:
    """One (run, test) pair's inputs: synthesized from `fixture_prng`, and checksummed.

    Seeds are namespaced by run and test name, so every run's and every test's tensors are
    independent -- adding or removing a test, or changing `RUNS`, never moves another
    tensor's bytes. `test_kernels.rs` builds the identical names.
    """

    def __init__(self, test: str, run: int):
        self.test = test
        self.run = run
        self.checks: dict[str, int] = {}

    def _seed(self, name: str) -> str:
        return f"run{self.run}.{self.test}.{name}"

    def _keep(self, name: str, storage: np.ndarray) -> None:
        self.checks[name] = prng.checksum(storage)

    def bf16(self, name, shape, span) -> torch.Tensor:
        count = int(np.prod(shape))
        bits = prng.bf16_uniform(self._seed(name), count, span[0], span[1])
        self._keep(name, bits)
        return _tensor(bits, torch.bfloat16, shape)

    def signs(self, name, shape, scale: float = 1.0) -> torch.Tensor:
        count = int(np.prod(shape))
        bits = prng.bf16_signs(self._seed(name), count, scale)
        self._keep(name, bits)
        return _tensor(bits, torch.bfloat16, shape)

    def derived(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Checksum a bf16 tensor this generator *computed* rather than drew.

        Every other input here is synthesized bit-for-bit from `fixture_prng`, so
        checksumming the draw is enough. A derived input also depends on arithmetic --
        for `x_exact` that is an f32 divide and one bf16 rounding -- which `test_kernels`
        has to reproduce independently. Checksumming only the operands would let a
        divergence in that arithmetic through, and it would surface as a plausible
        numeric error in the projection tests: exactly what these checksums exist to keep
        from happening.
        """
        bits = value.contiguous().view(torch.int16).numpy().view(np.uint16)
        self._keep(name, bits)
        return value

    def f8(self, name, shape, band, signed: bool = True) -> torch.Tensor:
        """f8e4m3 codes; returns their exact float values (the widening is lossless)."""
        count = int(np.prod(shape))
        codes = prng.f8_banded(self._seed(name), count, band[0], band[1], signed)
        self._keep(name, codes)
        return _tensor(codes, torch.float8_e4m3fn, shape).float()

    def f4(self, name, shape) -> torch.Tensor:
        """NVFP4 codes packed two per byte; returns their decoded float values."""
        count = int(np.prod(shape))
        packed = prng.f4_nibbles(self._seed(name), count)
        self._keep(name, packed)
        codes = torch.from_numpy(
            np.stack([packed & 0x0F, packed >> 4], axis=-1).reshape(-1).astype(np.int64)
        )
        values = F4_MAGNITUDES[codes & 0x7] * torch.where(codes >= 8, -1.0, 1.0)
        return values.reshape(shape)

    def constant_f32(self, name, values: np.ndarray) -> torch.Tensor:
        """A deterministic f32 operand -- masks, global scales. Still checksummed: the
        Rust side computes it rather than reading it, so it can still disagree."""
        values = np.ascontiguousarray(values, dtype=np.float32)
        self._keep(name, values)
        return _tensor(values, torch.float32, values.shape)

    def constant_bf16(self, name, values: np.ndarray) -> torch.Tensor:
        """A deterministic bf16 operand -- RoPE tables, the per-layer gate."""
        bits = prng.f32_to_bf16_bits(np.ascontiguousarray(values, dtype=np.float32))
        self._keep(name, bits)
        return _tensor(bits, torch.bfloat16, np.shape(values))

    def constant_i32(self, name, value: int) -> None:
        """`kv_offset`: consumed only by the kernel, so nothing is returned."""
        self._keep(name, np.array([value], dtype=np.int32))



def rope_tables(head_dim: int, theta: float, partial_rotary_factor: float, pos: int):
    """cos/sin for one position, computed in f64 then narrowed f64 -> f32 -> bf16.

    `gemma4.precompute_rope` computes these in f32. We use f64 instead so that
    `test_kernels.rs` can reproduce the table without having to match torch's f32
    `powf`/`cosf` bit-for-bit -- at f32 a last-ulp libm difference has a ~2**-16 chance
    of moving the bf16 result, at f64 it is ~2**-45. `verify_rope_matches_upstream`
    asserts the two agree exactly in bf16, so the extra precision costs no fidelity.
    """
    angles = int(partial_rotary_factor * head_dim // 2)
    inv_freq = 1.0 / (np.float64(theta) ** (np.arange(0, 2 * angles, 2, dtype=np.float64) / head_dim))
    inv_freq = np.concatenate([inv_freq, np.zeros(head_dim // 2 - angles, dtype=np.float64)])
    emb = np.concatenate([pos * inv_freq, pos * inv_freq])
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def negate_low_half(sin: np.ndarray) -> np.ndarray:
    """What the kernels actually receive as `sin`.

    The device RoPE spells `rotate_half` as a plain half-swap with no negation and folds
    the sign into `sin` instead, so the low half arrives pre-negated. The *reference*
    uses this file's own `apply_rotary_pos_emb` (built on `gemma4.rotate_half`) with the
    unmodified `sin`; only the kernel operand is transformed.
    """
    out = sin.copy()
    out[: len(out) // 2] *= -1.0
    return out


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
    """Single-tensor RoPE application (called once for q, once for k). The rotation
    itself is `gemma4.rotate_half` -- this repo's own from-scratch reference, not a
    reimplementation here -- this wrapper only handles the broadcast-shape bookkeeping
    for the per-token tensors this file works with."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (gemma4.rotate_half(x) * sin)


_ROPE_SANITY_POS = 137
"""Position `verify_rope_matches_upstream` checks at -- fixed, unlike `_run_pos`'s.

This check's job is to catch `rope_tables()`'s formula drifting from
`gemma4.precompute_rope`'s, which is a property of the formula, not of any one position.
Tying it to a per-run position instead would make it fail intermittently: comparing our f64
table against `gemma4.precompute_rope`'s native f32 one is an f32-vs-f64 libm comparison,
which (per `rope_tables`'s docstring) has a real, if small, per-position chance of a
last-ulp disagreement landing on a bf16 rounding boundary. A fixed position keeps this a
deterministic regression check instead of a once-in-a-while flake.
"""


def verify_rope_matches_upstream() -> None:
    """Assert our f64 tables are bit-identical to `reference/gemma4.py`'s own f32 RoPE, in bf16."""
    cases = (
        ("sliding_attention", DS, 10_000.0, 1.0),
        ("full_attention", DF, 1_000_000.0, 0.25),
    )
    pos = _ROPE_SANITY_POS
    for layer_type, head_dim, theta, factor in cases:
        cos_up, sin_up = gemma4.precompute_rope(head_dim, pos + 1, theta, factor, device="cpu", dtype=torch.bfloat16)
        cos, sin = rope_tables(head_dim, theta, factor, pos)
        for label, ours, theirs in (("cos", cos, cos_up[pos]), ("sin", sin, sin_up[pos])):
            mine = torch.from_numpy(ours).to(torch.bfloat16)
            if not torch.equal(mine, theirs.to(torch.bfloat16)):
                raise AssertionError(
                    f"{layer_type} {label}: f64 RoPE table diverges from "
                    f"gemma4.precompute_rope -- its formula has changed"
                )



def rms_norm(dim: int, weight: torch.Tensor | None, eps: float = EPS) -> "gemma4.RMSNorm":
    """A real `gemma4.RMSNorm`, with our weight installed (or unweighted)."""
    norm = gemma4.RMSNorm(dim, eps=eps, with_scale=weight is not None)
    if weight is not None:
        with torch.no_grad():
            norm.weight.copy_(weight)
    return norm.to(torch.bfloat16)


def scaled_projection(x: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The checkpoint's f8-storage projection: contract in f32, then scale the result.

    Hand-rolled because there is no upstream module for it -- the stored weight is
    `f8e4m3` codes plus one bf16 scale per output channel, not an `nn.Linear` weight.
    This mirrors the kernel exactly: the weight tile is widened to bf16 and contracted
    against an unquantized bf16 activation, and the per-output-channel scale is applied
    afterwards to the f32 result. Weight-only 8-bit, not W8A8.
    """
    return ((x.float() @ codes.T) * scale.float()).to(torch.bfloat16)


class Capture(dict):
    """Intermediates recorded along a reference pass.

    One pass produces several kernels' expected values -- `sliding_project_qkv` yields
    q, k and v -- so the reference stashes them here rather than threading return tuples
    around.
    """

    def __call__(self, name: str, value: torch.Tensor) -> torch.Tensor:
        self[name] = value
        return value



def gen_sliding_project_qkv(run: int):
    """`ops::sliding_project_qkv`: input norm -> QKV -> q/k norm -> RoPE -> cache write.

    `x` is built as `signs / input_rms_weight` so the normalized activation is exactly
    representable, isolating semantic errors from bf16 drift.
    """
    s = Synth("sliding_project_qkv", run)
    keep = Capture()
    pos = _run_pos(run)

    input_rms_weight = s.bf16("input_rms_weight", (H,), RMS_WEIGHT)
    signs = s.signs("x_signs", (H,))
    x = s.derived("x_exact", (signs.float() / input_rms_weight.float()).to(torch.bfloat16))

    q_codes = s.f8("q_weight", (QS, H), WEIGHT_EXP)
    k_codes = s.f8("k_weight", (PS, H), WEIGHT_EXP)
    v_codes = s.f8("v_weight", (PS, H), WEIGHT_EXP)
    q_scale = s.bf16("q_weight_scale", (QS,), ROW_SCALE)
    k_scale = s.bf16("k_weight_scale", (PS,), ROW_SCALE)
    v_scale = s.bf16("v_weight_scale", (PS,), ROW_SCALE)
    q_rms_weight = s.bf16("q_rms_weight", (DS,), UNIT)
    k_rms_weight = s.bf16("k_rms_weight", (DS,), UNIT)

    cos_raw, sin_raw = rope_tables(DS, 10_000.0, 1.0, pos)
    cos = s.constant_bf16("cos", cos_raw).view(1, 1, DS)
    s.constant_bf16("sin", negate_low_half(sin_raw))
    sin = torch.from_numpy(sin_raw).to(torch.bfloat16).view(1, 1, DS)
    s.constant_i32("kv_offset", (pos % TS) * NS * DS * 2)
    s.constant_i32("rope_offset", pos * DS * 2)

    h = keep("normed", rms_norm(H, input_rms_weight)(x))

    q = rms_norm(DS, q_rms_weight)(scaled_projection(h, q_codes, q_scale).view(NS * GS, DS))
    q = apply_rotary_pos_emb(q.view(1, 1, NS * GS, DS), cos, sin, unsqueeze_dim=2)
    keep("q", q.view(NS, GS, DS))

    k_raw = scaled_projection(h, k_codes, k_scale).view(NS, DS)
    k = rms_norm(DS, k_rms_weight)(k_raw)
    keep("k", apply_rotary_pos_emb(k.view(1, 1, NS, DS), cos, sin, unsqueeze_dim=2).view(NS, DS))

    v_raw = scaled_projection(h, v_codes, v_scale).view(NS, DS)
    keep("v", rms_norm(DS, None)(v_raw))

    return {"expected.q": keep["q"], "expected.k": keep["k"], "expected.v": keep["v"]}, s.checks


def gen_sliding_attention_output(run: int):
    """`ops::sliding_attention_output`: o_proj -> post-attention RMSNorm -> residual.

    `x` stands in for the attention head's real output, so it must be continuously
    valued like a real activation. It used to be drawn from `signs()` (always exactly
    `bf16(+1)`/`bf16(-1)`), which let `project_output`'s matmul get away with a
    sign-magnitude fast path that is exact for `+/-1` operands but wrong in general --
    the test never exercised anything else.
    """
    s = Synth("sliding_attention_output", run)
    x = s.bf16("x", (NS, GS, DS), ACTIVATION)
    post_rms_weight = s.bf16("post_attn_rms_weight", (H,), UNIT)
    o_codes = s.f8("o_weight", (H, QS), WEIGHT_EXP)
    o_scale = s.bf16("o_weight_scale", (H,), ROW_SCALE)
    residual = s.bf16("residual", (H,), UNIT)

    projected = scaled_projection(x.reshape(NS * GS * DS), o_codes, o_scale)
    normalized = rms_norm(H, post_rms_weight)(projected)
    return {"expected": (residual + normalized).to(torch.bfloat16)}, s.checks


class GloballyScaled(torch.nn.Module):
    """A real `nn.Linear` followed by the NVFP4 global-scale commit.

    The kernel contracts the locally-scaled weights, then applies the reciprocal global
    scale in f32 and commits to bf16. Keeping that boundary explicit around `gemma4.MLP`'s
    plain `nn.Linear` is the only adapter the MLP reference needs.
    """

    def __init__(self, linear: torch.nn.Linear, scale: float):
        super().__init__()
        self.linear = linear
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.linear(x).float() * self.scale).to(x.dtype)


def gen_decoder_feedforward(run: int):
    """`ops::decoder_feedforward`: pre-ff norm -> NVFP4 MLP -> post-ff norm -> residual
    -> per-layer gate.

    The MLP is `gemma4.MLP` (so the GeGLU shape and `gelu_pytorch_tanh` come from this
    repo's own checkpoint-matched reference) with the dequantized NVFP4 weights
    installed. Two levels of scale: an f8 per-16-element local scale folded into the
    weight here, and one f32 global scale per matrix applied as its reciprocal after the
    contraction.
    """
    s = Synth("decoder_feedforward", run)
    cfg = gemma4.Gemma4Config()

    residual = s.bf16("residual", (H,), UNIT)
    pre_ff_rms_weight = s.bf16("pre_ff_rms_weight", (H,), UNIT)
    post_ff_rms_weight = s.bf16("post_ff_rms_weight", (H,), UNIT)
    layer_scalar = s.constant_bf16("layer_scalar", np.full(8, LAYER_SCALAR, dtype=np.float32))

    weights = {}
    raw_scales = {}
    for name, out_dim, in_dim in (("up", L, H), ("gate", L, H), ("down", H, L)):
        codes = s.f4(f"{name}_weight_packed", (out_dim, in_dim))
        local = s.f8(f"{name}_weight_scale", (out_dim, in_dim // 16), LOCAL_SCALE_EXP, signed=False)
        weights[name] = codes * local.repeat_interleave(16, dim=-1)
        raw_scales[name] = float(
            prng.f32_uniform(f"run{run}.global.{name}_global_scale", 1, GLOBAL_SCALE[0], GLOBAL_SCALE[1])[0]
        )
        s.constant_f32(f"{name}_global_scale", np.array([1.0 / raw_scales[name]], dtype=np.float32))

    mlp = gemma4.MLP(cfg).to(torch.bfloat16)
    with torch.no_grad():
        for name in ("up", "gate", "down"):
            getattr(mlp, f"{name}_proj").weight.copy_(weights[name].to(torch.bfloat16))
        for name in ("up", "gate", "down"):
            setattr(mlp, f"{name}_proj", GloballyScaled(getattr(mlp, f"{name}_proj"), 1.0 / raw_scales[name]))

        hidden = rms_norm(H, pre_ff_rms_weight)(residual)
        hidden = rms_norm(H, post_ff_rms_weight)(mlp(hidden))
        expected = ((residual + hidden).to(torch.bfloat16) * layer_scalar[0]).to(torch.bfloat16)
    return {"expected": expected}, s.checks


TESTS = {
    "sliding_project_qkv": gen_sliding_project_qkv,
    "sliding_attention_output": gen_sliding_attention_output,
    "decoder_feedforward": gen_decoder_feedforward,
}


def generate() -> None:
    torch.set_grad_enabled(False)
    verify_rope_matches_upstream()

    entries: dict[str, torch.Tensor] = {}
    for run in range(RUNS):
        print(f"run {run}:")
        for name, build in TESTS.items():
            outputs, checks = build(run)
            for label, tensor in outputs.items():
                entries[f"run{run}.{name}.{label}"] = tensor.detach().float().contiguous()
            for label, value in checks.items():
                entries[f"run{run}.{name}.check.{label}"] = torch.tensor([value], dtype=torch.int64)
            print(f"  {name:34} {', '.join(outputs)}  ({len(checks)} inputs)")

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(entries, str(FIXTURE))
    size = FIXTURE.stat().st_size
    print(f"\nwrote {FIXTURE.relative_to(CRATE)} ({size / 1e6:.2f} MB, {RUNS} runs x {len(TESTS)} tests)")


if __name__ == "__main__":
    generate()

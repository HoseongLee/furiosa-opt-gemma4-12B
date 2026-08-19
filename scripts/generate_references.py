#!/usr/bin/env python3
"""Generate `ref/fixtures.safetensors`: the expected output of every device kernel.

`src/bin/test_kernels.rs` replays these on hardware and compares. This is now the only
kernel-test path in the crate -- an earlier `scripts/test.py` drove the same kernels
through the `furiosa.torch` VISA bridge as an independent second implementation, but that
bridge could not upload NVFP4 parameters (so `decoder_feedforward` was permanently
skipped) and could not be shipped to a remote NPU. This path needs no bridge, covers all
16 kernels, and is the one `scripts/rngd_test.sh` submits.

**The reference is the real model.** Everything here is built from
`transformers.models.gemma4_unified`, the architecture the checkpoint actually declares
(`config.json` -> `Gemma4UnifiedForConditionalGeneration`). Earlier revisions of this file
used the *generic* `transformers.models.gemma4` modules, which are a different model.
Where a real module exists it is instantiated and its submodules are called directly:
`Gemma4UnifiedRMSNorm`, `Gemma4UnifiedTextMLP`, `Gemma4UnifiedVisionEmbedder`,
`Gemma4UnifiedMultimodalEmbedder`, `Gemma4UnifiedTextScaledWordEmbedding` and
`apply_rotary_pos_emb` are all upstream code, not reimplementations.

Submodules are called individually rather than through a whole `forward` because the
kernel split is finer than the module split -- `Gemma4UnifiedTextAttention.forward` is
three of our kernels, and the boundaries between them are exactly what we need to
compare. Each reference records its intermediates into a `Capture` dict so one pass can
yield several kernels' expected values. Only the parts under test are ever run; nothing
here executes 48 layers.

Three things genuinely have no upstream module to call, and say so where they appear:
the masked/paged attention itself (upstream's is entangled with `Cache`), the f8-storage
projections (the checkpoint's weight form is not a plain `nn.Linear`), and the NVFP4 MLP
weights.

**Inputs are not stored.** Every kernel input is synthesized from `fixture_prng`, which
`test_kernels.rs` reimplements byte-for-byte, so the fixture carries only expected
outputs and stays around a megabyte rather than the ~2.3 GB the inputs would need. A
checksum of every synthesized input travels with them so a divergence between the two
implementations fails by name instead of as a mysterious numeric error.

No checkpoint is required: the config values below are literals, and the three NVFP4
global scales are the ones read from layer 0 of the real checkpoint.

Adding a test: write a `gen_*` function that returns `(outputs, checksums)`, add it to
`TESTS`, and add the matching shim and tolerance row in `src/bin/test_kernels.rs`. The two
registries are keyed by the same name and are kept in step by hand.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixture_prng as prng

from transformers.models.gemma4_unified.configuration_gemma4_unified import (
    Gemma4UnifiedAudioConfig,
    Gemma4UnifiedTextConfig,
    Gemma4UnifiedVisionConfig,
)
from transformers.models.gemma4_unified.modeling_gemma4_unified import (
    Gemma4UnifiedMultimodalEmbedder,
    Gemma4UnifiedRMSNorm,
    Gemma4UnifiedTextMLP,
    Gemma4UnifiedTextRotaryEmbedding,
    Gemma4UnifiedTextScaledWordEmbedding,
    Gemma4UnifiedVisionEmbedder,
    apply_rotary_pos_emb,
)

CRATE = Path(__file__).resolve().parent.parent
FIXTURE = CRATE / "ref" / "fixtures.safetensors"

H, L, W = 3840, 15360, 262144
NS, GS, DS, QS, PS, TS = 8, 2, 256, 4096, 2048, 1024
GF, DF, QF, PF, TF = 16, 512, 8192, 512, 512
RV, MV, OV, AA = 6912, 3840, 3840, 640

EMBED_SCALE = 61.967_734
LOGIT_SOFTCAP = 30.0
EPS = 1e-6

VISION_EPS = 1e-5

RAW_GLOBAL_SCALES = {"up": 9600.0, "gate": 9600.0, "down": 12928.0}

F4_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

WEIGHT_EXP = (7, 14)
LOCAL_SCALE_EXP = (8, 10)
ROW_SCALE = (0.0, 1e-3)
UNIT = (0.0, 1.0)
CENTRED = (-0.5, 0.5)
RMS_WEIGHT = (0.75, 1.25)

POS = 137
SLIDING_VALID = 300
FULL_ROW = 137
LAYER_SCALAR = 0.375
EMBED_TOKEN_ID = 90210


def text_config() -> Gemma4UnifiedTextConfig:
    """The checkpoint's `text_config`, as literals -- no checkpoint on disk required."""
    return Gemma4UnifiedTextConfig(
        hidden_size=H,
        intermediate_size=L,
        vocab_size=W,
        num_attention_heads=16,
        num_key_value_heads=NS,
        num_global_key_value_heads=1,
        head_dim=DS,
        global_head_dim=DF,
        num_hidden_layers=48,
        sliding_window=TS,
        rms_norm_eps=EPS,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        attention_dropout=0.0,
        attention_k_eq_v=True,
        final_logit_softcapping=LOGIT_SOFTCAP,
        use_double_wide_mlp=False,
        enable_moe_block=False,
        num_kv_shared_layers=0,
        tie_word_embeddings=True,
        layer_types=["full_attention" if i % 6 == 5 else "sliding_attention" for i in range(48)],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1_000_000.0,
                "partial_rotary_factor": 0.25,
            },
        },
    )


def vision_config() -> Gemma4UnifiedVisionConfig:
    return Gemma4UnifiedVisionConfig(
        mm_embed_dim=MV, mm_posemb_size=1120, output_proj_dims=OV, num_soft_tokens=280,
        patch_size=16, pooling_kernel_size=3, rms_norm_eps=EPS,
    )


def audio_config() -> Gemma4UnifiedAudioConfig:
    return Gemma4UnifiedAudioConfig(audio_embed_dim=AA, output_proj_dims=AA, rms_norm_eps=EPS)



def _tensor(storage: np.ndarray, dtype: torch.dtype, shape) -> torch.Tensor:
    """Reinterpret raw storage bytes as a torch tensor, without any float conversion."""
    return torch.frombuffer(bytearray(storage.tobytes()), dtype=dtype).reshape(shape)


class Synth:
    """One test's inputs: synthesized from `fixture_prng`, and checksummed.

    Seeds are namespaced by test name, so every test's tensors are independent and
    adding or removing a test never moves another's bytes. `test_kernels.rs` builds the
    identical names.
    """

    def __init__(self, test: str):
        self.test = test
        self.checks: dict[str, int] = {}

    def _seed(self, name: str) -> str:
        return f"{self.test}.{name}"

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

    def f32(self, name, shape, span) -> torch.Tensor:
        count = int(np.prod(shape))
        values = prng.f32_uniform(self._seed(name), count, span[0], span[1])
        self._keep(name, values)
        return _tensor(values, torch.float32, shape)

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

    Upstream computes these in f32 (`Gemma4UnifiedTextRotaryEmbedding.forward` forces
    float32). We use f64 instead so that `test_kernels.rs` can reproduce the table
    without having to match torch's f32 `powf`/`cosf` bit-for-bit -- at f32 a last-ulp
    libm difference has a ~2**-16 chance of moving the bf16 result, at f64 it is ~2**-45.
    `verify_rope_matches_upstream` asserts the two agree exactly in bf16, so the extra
    precision costs no fidelity.
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
    uses upstream's `apply_rotary_pos_emb` with the unmodified `sin`; only the kernel
    operand is transformed.
    """
    out = sin.copy()
    out[: len(out) // 2] *= -1.0
    return out


def verify_rope_matches_upstream(config: Gemma4UnifiedTextConfig) -> None:
    """Assert our f64 tables are bit-identical to the real rotary module's, in bf16."""
    rotary = Gemma4UnifiedTextRotaryEmbedding(config)
    cases = (
        ("sliding_attention", DS, 10_000.0, 1.0),
        ("full_attention", DF, 1_000_000.0, 0.25),
    )
    for layer_type, head_dim, theta, factor in cases:
        probe = torch.zeros(1, 1, head_dim, dtype=torch.bfloat16)
        cos_up, sin_up = rotary(probe, torch.tensor([[POS]]), layer_type=layer_type)
        cos, sin = rope_tables(head_dim, theta, factor, POS)
        for label, ours, theirs in (("cos", cos, cos_up), ("sin", sin, sin_up)):
            mine = torch.from_numpy(ours).to(torch.bfloat16)
            if not torch.equal(mine, theirs[0, 0].to(torch.bfloat16)):
                raise AssertionError(
                    f"{layer_type} {label}: f64 RoPE table diverges from "
                    f"Gemma4UnifiedTextRotaryEmbedding -- upstream's formula has changed"
                )



def rms_norm(dim: int, weight: torch.Tensor | None, eps: float = EPS) -> Gemma4UnifiedRMSNorm:
    """A real `Gemma4UnifiedRMSNorm`, with our weight installed (or unweighted)."""
    norm = Gemma4UnifiedRMSNorm(dim, eps=eps, with_scale=weight is not None)
    if weight is not None:
        with torch.no_grad():
            norm.weight.copy_(weight)
    return norm.to(torch.bfloat16)


def layer_norm(dim: int, weight: torch.Tensor, bias: torch.Tensor) -> torch.nn.LayerNorm:
    """A real `nn.LayerNorm`, matching the vision embedder's construction (default eps)."""
    norm = torch.nn.LayerNorm(dim, eps=VISION_EPS)
    with torch.no_grad():
        norm.weight.copy_(weight)
        norm.bias.copy_(bias)
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


def masked_attention(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Boolean-masked softmax with scale 1.0 (Gemma folds the scale into the norms).

    Hand-rolled: upstream's attention is entangled with `Cache` and its sliding-window
    bookkeeping, while these kernels take a plain mask and a ring/page of KV.
    """
    return torch.softmax(scores.masked_fill(~mask.bool().unsqueeze(0), float("-inf")), dim=-1)


class Capture(dict):
    """Intermediates recorded along a reference pass.

    One pass produces several kernels' expected values -- `sliding_project_qkv` yields
    q, k and v -- so the reference stashes them here rather than threading return tuples
    around.
    """

    def __call__(self, name: str, value: torch.Tensor) -> torch.Tensor:
        self[name] = value
        return value



def gen_embed_token():
    """`ops::embed_token`: gather one row out of the `[W, H]` table, then scale it by
    EMBED_SCALE.

    The gather is now the kernel's own job (`embedding_table.dma_gather_scaled(offset)`),
    so `test_kernels.rs` builds a full `[W, H]` table with every row but `EMBED_TOKEN_ID`
    left zero. That placement is a test-harness detail the reference does not need to
    reproduce -- `offset` is only checksummed so the two sides agree on the byte-offset
    formula, the same way `kv_offset`/`rope_offset` are checked elsewhere. The real
    `Gemma4UnifiedTextScaledWordEmbedding` is still built with a two-row table and index 1
    read -- row 0 is the `padding_idx` and is forced to zero.
    """
    s = Synth("embed_token")
    row = s.bf16("row", (H,), CENTRED)
    s.constant_i32("offset", EMBED_TOKEN_ID * H * 2)

    embedding = Gemma4UnifiedTextScaledWordEmbedding(2, H, padding_idx=0, embed_scale=EMBED_SCALE)
    embedding = embedding.to(torch.bfloat16)
    with torch.no_grad():
        embedding.weight[1].copy_(row)
        expected = embedding(torch.tensor([1]))[0]
    return {"expected": expected}, s.checks


def gen_copy_hidden_state():
    """`ops::copy_hidden_state`: an unscaled copy -- a soft token replaces a text
    embedding outright, so EMBED_SCALE is deliberately skipped."""
    s = Synth("copy_hidden_state")
    x = s.bf16("x", (H,), CENTRED)
    return {"expected": x}, s.checks


def _project_sliding(label: str, exact: bool):
    """`ops::sliding_project_qkv`: input norm -> QKV -> q/k norm -> RoPE -> cache write.

    `exact` builds `x` as `signs / input_rms_weight` so the normalized activation is
    exactly representable, isolating semantic errors from bf16 drift; the random variant
    uses independent values and measures deployment error instead.
    """
    s = Synth(label)
    keep = Capture()

    input_rms_weight = s.bf16("input_rms_weight", (H,), RMS_WEIGHT)
    signs = s.signs("x_signs", (H,))
    if exact:
        x = s.derived("x_exact", (signs.float() / input_rms_weight.float()).to(torch.bfloat16))
    else:
        x = signs

    q_codes = s.f8("q_weight", (QS, H), WEIGHT_EXP)
    k_codes = s.f8("k_weight", (PS, H), WEIGHT_EXP)
    v_codes = s.f8("v_weight", (PS, H), WEIGHT_EXP)
    q_scale = s.bf16("q_weight_scale", (QS,), ROW_SCALE)
    k_scale = s.bf16("k_weight_scale", (PS,), ROW_SCALE)
    v_scale = s.bf16("v_weight_scale", (PS,), ROW_SCALE)
    q_rms_weight = s.bf16("q_rms_weight", (DS,), UNIT)
    k_rms_weight = s.bf16("k_rms_weight", (DS,), UNIT)

    cos_raw, sin_raw = rope_tables(DS, 10_000.0, 1.0, POS)
    cos = s.constant_bf16("cos", cos_raw).view(1, 1, DS)
    s.constant_bf16("sin", negate_low_half(sin_raw))
    sin = torch.from_numpy(sin_raw).to(torch.bfloat16).view(1, 1, DS)
    s.constant_i32("kv_offset", (POS % TS) * NS * DS * 2)
    s.constant_i32("rope_offset", POS * DS * 2)

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


def gen_sliding_project_qkv_exact():
    return _project_sliding("sliding_project_qkv_exact", exact=True)


def gen_sliding_project_qkv_random():
    return _project_sliding("sliding_project_qkv_random", exact=False)


def gen_sliding_attention():
    """`ops::sliding_attention`: one decode query against the whole Ts-slot ring.

    q is `+/-1/16` and k is `+/-1` so the score reduction is bf16-exact and the softmax
    does not saturate -- the test then measures attention, not score quantization.
    """
    s = Synth("sliding_attention")
    q = s.signs("q", (NS, GS, DS), 1 / 16)
    k = s.signs("k", (TS, NS, DS))
    v = s.bf16("v", (TS, NS, DS), CENTRED)
    mask_values = np.zeros(TS, dtype=np.float32)
    mask_values[:SLIDING_VALID] = 1.0
    mask = s.constant_f32("mask", mask_values)

    q_heads = q.float().reshape(NS * GS, DS)
    k_heads = k.float().permute(1, 0, 2).repeat_interleave(GS, dim=0)
    v_heads = v.float().permute(1, 0, 2).repeat_interleave(GS, dim=0)
    weights = masked_attention(torch.einsum("hd,htd->ht", q_heads, k_heads), mask)
    expected = torch.einsum("ht,htd->hd", weights, v_heads).reshape(NS, GS, DS)
    return {"expected": expected.to(torch.bfloat16)}, s.checks


def _attention_output_sliding(label: str, exact: bool):
    """`ops::sliding_attention_output`: o_proj -> post-attention RMSNorm -> residual."""
    s = Synth(label)
    x = s.signs("x", (NS, GS, DS)) if exact else s.bf16("x", (NS, GS, DS), CENTRED)
    post_rms_weight = s.bf16("post_attn_rms_weight", (H,), UNIT)
    o_codes = s.f8("o_weight", (H, QS), WEIGHT_EXP)
    o_scale = s.bf16("o_weight_scale", (H,), ROW_SCALE)
    residual = s.bf16("residual", (H,), UNIT)

    projected = scaled_projection(x.reshape(NS * GS * DS), o_codes, o_scale)
    normalized = rms_norm(H, post_rms_weight)(projected)
    return {"expected": (residual + normalized).to(torch.bfloat16)}, s.checks


def gen_sliding_attention_output_exact():
    return _attention_output_sliding("sliding_attention_output_exact", exact=True)


def gen_sliding_attention_output_random():
    return _attention_output_sliding("sliding_attention_output_random", exact=False)


def _project_full(label: str, exact: bool):
    """`ops::full_project_qkv`: true MQA -- one KV head, and no v_proj at all.

    V is derived from K (`v_norm(k_raw)`, unweighted), which is what upstream does when
    `attention_k_eq_v` is set and the layer is not sliding. Deliberate, not a stub.
    """
    s = Synth(label)
    keep = Capture()

    input_rms_weight = s.bf16("input_rms_weight", (H,), RMS_WEIGHT)
    signs = s.signs("x_signs", (H,))
    x = (
        s.derived("x_exact", (signs.float() / input_rms_weight.float()).to(torch.bfloat16))
        if exact
        else signs
    )

    q_codes = s.f8("q_weight", (QF, H), WEIGHT_EXP)
    k_codes = s.f8("k_weight", (PF, H), WEIGHT_EXP)
    q_scale = s.bf16("q_weight_scale", (QF,), ROW_SCALE)
    k_scale = s.bf16("k_weight_scale", (PF,), ROW_SCALE)
    q_rms_weight = s.bf16("q_rms_weight", (DF,), UNIT)
    k_rms_weight = s.bf16("k_rms_weight", (DF,), UNIT)

    cos_raw, sin_raw = rope_tables(DF, 1_000_000.0, 0.25, POS)
    cos = s.constant_bf16("cos", cos_raw).view(1, 1, DF)
    s.constant_bf16("sin", negate_low_half(sin_raw))
    sin = torch.from_numpy(sin_raw).to(torch.bfloat16).view(1, 1, DF)
    s.constant_i32("kv_offset", POS * DF * 2)
    s.constant_i32("rope_offset", POS * DF * 2)

    h = keep("normed", rms_norm(H, input_rms_weight)(x))

    q = rms_norm(DF, q_rms_weight)(scaled_projection(h, q_codes, q_scale).view(GF, DF))
    keep("q", apply_rotary_pos_emb(q.view(1, 1, GF, DF), cos, sin, unsqueeze_dim=2).view(GF, DF))

    k_raw = scaled_projection(h, k_codes, k_scale).view(DF)
    k = rms_norm(DF, k_rms_weight)(k_raw)
    keep("k", apply_rotary_pos_emb(k.view(1, 1, 1, DF), cos, sin, unsqueeze_dim=2).view(DF))
    keep("v", rms_norm(DF, None)(k_raw))

    return {"expected.q": keep["q"], "expected.k": keep["k"], "expected.v": keep["v"]}, s.checks


def gen_full_project_qkv_exact():
    return _project_full("full_project_qkv_exact", exact=True)


def gen_full_project_qkv_random():
    return _project_full("full_project_qkv_random", exact=False)


def gen_full_attention():
    """`ops::full_attention_first_page` then `ops::full_attention_page`.

    Two Tf-wide pages: page 0 already resident and fully valid, page 1 causal up to the
    live query's row. The kernels accumulate a running max/sum/output across pages
    (flash-style online softmax) and leave the output unnormalized; `test_kernels` divides
    by the running sum, isolating page chaining from the output pipeline.
    """
    s = Synth("full_attention")
    q = s.signs("q", (GF, DF), 1 / 16)
    k0 = s.signs("k0", (TF, DF))
    k1 = s.signs("k1", (TF, DF))
    v0 = s.bf16("v0", (TF, DF), CENTRED)
    v1 = s.bf16("v1", (TF, DF), CENTRED)
    mask0 = s.constant_f32("mask0", np.ones(TF, dtype=np.float32))
    page1 = np.zeros(TF, dtype=np.float32)
    page1[: FULL_ROW + 1] = 1.0
    mask1 = s.constant_f32("mask1", page1)

    k = torch.cat([k0, k1]).float()
    v = torch.cat([v0, v1]).float()
    mask = torch.cat([mask0, mask1])
    weights = masked_attention(torch.einsum("gd,td->gt", q.float(), k), mask)
    expected = torch.einsum("gt,td->gd", weights, v)
    return {"expected": expected.to(torch.bfloat16)}, s.checks


def _attention_output_full(label: str, exact: bool):
    """`ops::full_attention_output`: normalize by the running sum, then o_proj -> norm ->
    residual. The softmax divide the paged kernels deferred happens here."""
    s = Synth(label)
    x = s.signs("x", (GF, DF)) if exact else s.bf16("x", (GF, DF), CENTRED)
    running_sum = (
        s.constant_f32("running_sum", np.full(GF, 4.0, dtype=np.float32))
        if exact
        else s.f32("running_sum", (GF,), (2.0, 4.0))
    )
    post_rms_weight = s.bf16("post_attn_rms_weight", (H,), UNIT)
    o_codes = s.f8("o_weight", (H, QF), WEIGHT_EXP)
    o_scale = s.bf16("o_weight_scale", (H,), ROW_SCALE)
    residual = s.bf16("residual", (H,), UNIT)

    normalized_x = (x.float() / running_sum.unsqueeze(-1)).to(torch.bfloat16)
    projected = scaled_projection(normalized_x.reshape(GF * DF), o_codes, o_scale)
    normalized = rms_norm(H, post_rms_weight)(projected)
    return {"expected": (residual + normalized).to(torch.bfloat16)}, s.checks


def gen_full_attention_output_exact():
    return _attention_output_full("full_attention_output_exact", exact=True)


def gen_full_attention_output_random():
    return _attention_output_full("full_attention_output_random", exact=False)


class GloballyScaled(torch.nn.Module):
    """A real `nn.Linear` followed by the NVFP4 global-scale commit.

    The kernel contracts the locally-scaled weights, then applies the reciprocal global
    scale in f32 and commits to bf16. Keeping that boundary explicit around the genuine
    Transformers linear is the only adapter the MLP reference needs.
    """

    def __init__(self, linear: torch.nn.Linear, scale: float):
        super().__init__()
        self.linear = linear
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.linear(x).float() * self.scale).to(x.dtype)


def gen_decoder_feedforward():
    """`ops::decoder_feedforward`: pre-ff norm -> NVFP4 MLP -> post-ff norm -> residual
    -> per-layer gate.

    The MLP is the real `Gemma4UnifiedTextMLP` (so the GeGLU shape and
    `gelu_pytorch_tanh` come from upstream) with the dequantized NVFP4 weights installed.
    Two levels of scale: an f8 per-16-element local scale folded into the weight here,
    and one f32 global scale per matrix applied as its reciprocal after the contraction.
    """
    s = Synth("decoder_feedforward")
    config = text_config()

    residual = s.bf16("residual", (H,), UNIT)
    pre_ff_rms_weight = s.bf16("pre_ff_rms_weight", (H,), UNIT)
    post_ff_rms_weight = s.bf16("post_ff_rms_weight", (H,), UNIT)
    layer_scalar = s.constant_bf16("layer_scalar", np.full(8, LAYER_SCALAR, dtype=np.float32))

    weights = {}
    for name, out_dim, in_dim in (("up", L, H), ("gate", L, H), ("down", H, L)):
        codes = s.f4(f"{name}_weight_packed", (out_dim, in_dim))
        local = s.f8(f"{name}_weight_scale", (out_dim, in_dim // 16), LOCAL_SCALE_EXP, signed=False)
        weights[name] = codes * local.repeat_interleave(16, dim=-1)
        s.constant_f32(f"{name}_global_scale", np.array([1.0 / RAW_GLOBAL_SCALES[name]], dtype=np.float32))

    mlp = Gemma4UnifiedTextMLP(config, layer_idx=0).to(torch.bfloat16)
    with torch.no_grad():
        for name in ("up", "gate", "down"):
            getattr(mlp, f"{name}_proj").weight.copy_(weights[name].to(torch.bfloat16))
        for name in ("up", "gate", "down"):
            setattr(mlp, f"{name}_proj", GloballyScaled(getattr(mlp, f"{name}_proj"), 1.0 / RAW_GLOBAL_SCALES[name]))

        hidden = rms_norm(H, pre_ff_rms_weight)(residual)
        hidden = rms_norm(H, post_ff_rms_weight)(mlp(hidden))
        expected = ((residual + hidden).to(torch.bfloat16) * layer_scalar[0]).to(torch.bfloat16)
    return {"expected": expected}, s.checks


def gen_final_norm_and_logits():
    """`ops::final_norm_and_logits`: final RMSNorm -> lm_head -> logit soft-cap.

    The lm_head is the production `[W, H]` shape -- 1.007e9 elements, 2 GB in bf16 -- so
    it is synthesized and contracted in row blocks and never materialized whole. The
    checksum is accumulated the same way, which is why `fixture_prng.checksum` takes a
    word offset.
    """
    s = Synth("final_norm_and_logits")
    x = s.bf16("x", (H,), UNIT)
    rms_weight = s.bf16("rms_weight", (H,), UNIT)

    normed = rms_norm(H, rms_weight)(x).float()
    logits = torch.empty(W, dtype=torch.float32)
    rows = max(1, (1 << 23) // H)
    total = 0
    for start in range(0, W, rows):
        count = min(rows, W - start)
        bits = prng.bf16_uniform(
            "final_norm_and_logits.lm_head_weight", count * H, CENTRED[0], CENTRED[1], offset=start * H
        )
        total = (total + prng.checksum(bits, word_offset=start * H // 2)) & 0xFFFF_FFFF
        block = _tensor(bits, torch.bfloat16, (count, H))
        logits[start : start + count] = (block.float() @ normed).to(torch.bfloat16).float()
    s.checks["lm_head_weight"] = total

    expected = (torch.tanh(logits / LOGIT_SOFTCAP) * LOGIT_SOFTCAP).to(torch.bfloat16)
    return {"expected": expected}, s.checks



def gen_patch_embed():
    """`ops_vision::patch_embed`: patch_ln1 -> patch_dense -> patch_ln2, one patch.

    Every vision weight is plain bf16 (config.json's `quantization_config.ignore` list),
    so unlike the text projections there is nothing quantized to construct around.
    """
    s = Synth("patch_embed")
    x = s.bf16("x", (RV,), CENTRED)
    ln1_weight = s.bf16("ln1_weight", (RV,), UNIT)
    ln1_bias = s.bf16("ln1_bias", (RV,), CENTRED)
    dense_weight = s.bf16("dense_weight", (MV, RV), CENTRED)
    dense_bias = s.bf16("dense_bias", (MV,), CENTRED)
    ln2_weight = s.bf16("ln2_weight", (MV,), UNIT)
    ln2_bias = s.bf16("ln2_bias", (MV,), CENTRED)

    embedder = Gemma4UnifiedVisionEmbedder(vision_config(), text_config()).to(torch.bfloat16)
    with torch.no_grad():
        embedder.patch_ln1 = layer_norm(RV, ln1_weight, ln1_bias)
        embedder.patch_ln2 = layer_norm(MV, ln2_weight, ln2_bias)
        embedder.patch_dense.weight.copy_(dense_weight)
        embedder.patch_dense.bias.copy_(dense_bias)
        hidden = embedder.patch_ln1(x)
        hidden = embedder.patch_dense(hidden)
        expected = embedder.patch_ln2(hidden)
    return {"expected": expected}, s.checks


def gen_add_position_and_norm():
    """`ops_vision::add_position_and_norm`: x + position embedding -> pos_norm.

    The position embedding is gathered and summed on the host (two table lookups, one
    per axis), so from the kernel's side it is just another Mv-wide tensor to add.
    """
    s = Synth("add_position_and_norm")
    x = s.bf16("x", (MV,), CENTRED)
    pos_embed = s.bf16("pos_embed", (MV,), CENTRED)
    ln3_weight = s.bf16("ln3_weight", (MV,), UNIT)
    ln3_bias = s.bf16("ln3_bias", (MV,), CENTRED)

    embedder = Gemma4UnifiedVisionEmbedder(vision_config(), text_config()).to(torch.bfloat16)
    with torch.no_grad():
        embedder.pos_norm = layer_norm(MV, ln3_weight, ln3_bias)
        expected = embedder.pos_norm((x + pos_embed).to(torch.bfloat16))
    return {"expected": expected}, s.checks


def gen_project_to_text_embedding():
    """`ops_vision::project_to_text_embedding`: unweighted RMSNorm -> bias-free projection
    into text space. The real `Gemma4UnifiedMultimodalEmbedder`, called as-is."""
    s = Synth("project_to_text_embedding")
    x = s.bf16("x", (OV,), CENTRED)
    proj_weight = s.bf16("proj_weight", (H, OV), CENTRED)

    embedder = Gemma4UnifiedMultimodalEmbedder(vision_config(), text_config()).to(torch.bfloat16)
    with torch.no_grad():
        embedder.embedding_projection.weight.copy_(proj_weight)
        expected = embedder(x)
    return {"expected": expected}, s.checks


def gen_audio_project_frame():
    """`ops_audio::audio_project_frame`: one 640-sample (40 ms) waveform frame.

    The checkpoint's audio embedder is encoder-free -- the same
    `Gemma4UnifiedMultimodalEmbedder` as the vision tail, built from `audio_config`, so
    the RMSNorm has no learned scale and the projection is a single bf16 [H, Aa] matrix.
    One frame is one soft token.
    """
    s = Synth("audio_project_frame")
    x = s.bf16("x", (AA,), CENTRED)
    proj_weight = s.bf16("proj_weight", (H, AA), CENTRED)

    embedder = Gemma4UnifiedMultimodalEmbedder(audio_config(), text_config()).to(torch.bfloat16)
    with torch.no_grad():
        embedder.embedding_projection.weight.copy_(proj_weight)
        expected = embedder(x)
    return {"expected": expected}, s.checks


def gen_layernorm():
    """`ops_vision::layernorm`: the standalone LayerNorm, exercised on its own rather than
    only inside patch_embed/add_position_and_norm, since it is its own kernel."""
    s = Synth("layernorm")
    x = s.bf16("x", (MV,), CENTRED)
    weight = s.bf16("weight", (MV,), UNIT)
    bias = s.bf16("bias", (MV,), CENTRED)
    return {"expected": layer_norm(MV, weight, bias)(x)}, s.checks


TESTS = {
    "embed_token": gen_embed_token,
    "copy_hidden_state": gen_copy_hidden_state,
    "sliding_project_qkv_exact": gen_sliding_project_qkv_exact,
    "sliding_project_qkv_random": gen_sliding_project_qkv_random,
    "sliding_attention": gen_sliding_attention,
    "sliding_attention_output_exact": gen_sliding_attention_output_exact,
    "sliding_attention_output_random": gen_sliding_attention_output_random,
    "full_project_qkv_exact": gen_full_project_qkv_exact,
    "full_project_qkv_random": gen_full_project_qkv_random,
    "full_attention": gen_full_attention,
    "full_attention_output_exact": gen_full_attention_output_exact,
    "full_attention_output_random": gen_full_attention_output_random,
    "decoder_feedforward": gen_decoder_feedforward,
    "final_norm_and_logits": gen_final_norm_and_logits,
    "patch_embed": gen_patch_embed,
    "add_position_and_norm": gen_add_position_and_norm,
    "project_to_text_embedding": gen_project_to_text_embedding,
    "audio_project_frame": gen_audio_project_frame,
    "layernorm": gen_layernorm,
}


def generate() -> None:
    torch.set_grad_enabled(False)
    verify_rope_matches_upstream(text_config())

    entries: dict[str, torch.Tensor] = {}
    for name, build in TESTS.items():
        outputs, checks = build()
        for label, tensor in outputs.items():
            entries[f"{name}.{label}"] = tensor.detach().float().contiguous()
        for label, value in checks.items():
            entries[f"{name}.check.{label}"] = torch.tensor([value], dtype=torch.int64)
        print(f"  {name:34} {', '.join(outputs)}  ({len(checks)} inputs)")

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    save_file(entries, str(FIXTURE))
    size = FIXTURE.stat().st_size
    print(f"\nwrote {FIXTURE.relative_to(CRATE)} ({size / 1e6:.2f} MB, {len(TESTS)} tests)")


if __name__ == "__main__":
    generate()

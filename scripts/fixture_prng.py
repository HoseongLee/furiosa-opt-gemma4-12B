"""Byte-exact tensor synthesis, shared with `src/bin/test_kernels.rs`.

The fixture that `generate_references.py` writes carries **only expected outputs**. Every
*input* a kernel test needs -- activations, weights, scales, masks, RoPE tables -- is
synthesized independently on both sides from this module's functions, and
`test_kernels.rs` implements exactly the same arithmetic. That is what keeps
`ref/fixtures.safetensors` around a megabyte instead of the ~2.3 GB the inputs would
occupy (the `[W, H]` lm_head weight alone is 2 GB).

Two properties make that safe to rely on:

- **Stateless addressing.** An element's value depends only on its tensor's *name* and
  its own index -- never on a running stream. Tests can be added, removed or reordered
  and no other tensor's bytes move.
- **No cross-language float conversion.** Values are built from raw bit patterns and the
  f32->bf16 rounding is spelled out here (`f32_to_bf16_bits`) rather than delegated to
  `torch`/`half`, so both sides run the identical algorithm rather than two
  implementations that are merely expected to agree.

Where a value genuinely cannot be built from bits -- the RoPE tables need `cos`/`sin` --
the computation runs in **f64** and is narrowed f64 -> f32 -> bf16. A last-ulp
disagreement between two libm implementations then has to also straddle a bf16 rounding
boundary to matter, which is ~2^-44 per element.

Belt and braces: `generate_references.py` records `checksum()` of every input it
synthesizes into the fixture, and `test_kernels.rs` verifies its own synthesis against
those before launching anything. If the two implementations ever diverge, the run fails
immediately with the offending tensor's name instead of surfacing as a plausible-looking
numeric error.
"""

import numpy as np

MASK64 = 0xFFFF_FFFF_FFFF_FFFF

_FNV_OFFSET = 0xCBF2_9CE4_8422_2325
_FNV_PRIME = 0x0000_0100_0000_01B3

_GOLDEN = np.uint64(0x9E37_79B9_7F4A_7C15)
_MIX_A = np.uint64(0xBF58_476D_1CE4_E5B9)
_MIX_B = np.uint64(0x94D0_49BB_1331_11EB)

CHUNK = 1 << 23


def name_hash(name: str) -> np.uint64:
    """FNV-1a over the tensor's name -- the per-tensor seed.

    Kept in Python ints with an explicit mask: numpy raises `RuntimeWarning: overflow
    encountered in scalar multiply` for wrapping uint64 *scalar* arithmetic, even though
    the wrap is exactly what FNV wants (array arithmetic wraps silently).
    """
    h = _FNV_OFFSET
    for byte in name.encode("utf-8"):
        h = ((h ^ byte) * _FNV_PRIME) & MASK64
    return np.uint64(h)


def words(name: str, count: int, offset: int = 0) -> np.ndarray:
    """`count` 64-bit words for `name`, starting at element `offset`.

    splitmix64 used as a mixing function over the element index rather than as a
    stateful generator, so element `i` is addressable without producing `0..i`.
    """
    index = np.arange(offset, offset + count, dtype=np.uint64)
    x = (name_hash(name) ^ index) + _GOLDEN
    x = (x ^ (x >> np.uint64(30))) * _MIX_A
    x = (x ^ (x >> np.uint64(27))) * _MIX_B
    return x ^ (x >> np.uint64(31))



def u01(w: np.ndarray) -> np.ndarray:
    """The top 24 bits as a float32 in [0, 1), exactly.

    24 bits is the f32 significand, so `k / 2**24` is representable with no rounding --
    the conversion is exact on both sides rather than merely close.
    """
    return (w >> np.uint64(40)).astype(np.uint32).astype(np.float32) / np.float32(16777216.0)


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """Round f32 to bf16 (round-half-to-even), returning the raw u16 bit patterns.

    Spelled out rather than delegated to `torch.Tensor.to(bfloat16)` so that this file
    and `test_kernels.rs` run the same algorithm instead of two conversions that are
    only expected to agree. Inputs here are always finite, so NaN/Inf need no special
    case.
    """
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) >> np.uint32(16)).astype(np.uint16)



def bf16_uniform(name: str, count: int, lo: float, hi: float, offset: int = 0) -> np.ndarray:
    """bf16 values uniform in [lo, hi).

    The affine step is `lo + (hi - lo) * u` in f32 -- one multiply then one add, in that
    order, matching the Rust side. Rust never contracts this into an FMA on its own.
    """
    u = u01(words(name, count, offset))
    lo32, hi32 = np.float32(lo), np.float32(hi)
    return f32_to_bf16_bits(lo32 + (hi32 - lo32) * u)


def f32_uniform(name: str, count: int, lo: float, hi: float, offset: int = 0) -> np.ndarray:
    """f32 values uniform in [lo, hi), stored as f32 -- no bf16 narrowing.

    For the genuinely-f32 kernel operands: `full_attention_output`'s `running_sum`.
    """
    u = u01(words(name, count, offset))
    lo32, hi32 = np.float32(lo), np.float32(hi)
    return lo32 + (hi32 - lo32) * u


def bf16_signs(name: str, count: int, scale: float = 1.0, offset: int = 0) -> np.ndarray:
    """bf16 `+/-scale`, sign from the top bit.

    `scale` exists for the attention tests, which use `+/-1/16` queries so that the
    score reduction stays bf16-exact and the softmax does not saturate -- that is what
    makes those tests measure attention rather than score quantization. Exact for a
    power-of-two `scale`, which is the only kind used.
    """
    magnitude = f32_to_bf16_bits(np.array([abs(scale)], dtype=np.float32))[0]
    negative = words(name, count, offset) >> np.uint64(63)
    return np.where(negative == 1, magnitude | np.uint16(0x8000), magnitude).astype(np.uint16)


def f8_banded(
    name: str, count: int, exp_min: int, exp_max: int, signed: bool = True, offset: int = 0
) -> np.ndarray:
    """f8e4m3 code bytes with the exponent field drawn from `[exp_min, exp_max]`.

    Built from bit fields rather than by rounding a float, so no f32->f8 conversion has
    to agree across two languages. The band is what makes the values *useful*: drawing
    codes uniformly would be log-uniform in magnitude and pile up near zero, whereas
    `test.py`'s `f8_weight` draws uniform in [-448, 448]. Picking an exponent band
    reproduces a comparable magnitude range while staying exact by construction.

    Value is `+/-2**(exp - 7) * (1 + mantissa/8)`. Keeping `exp_max <= 14` stays clear of
    e4m3fn's only non-finite pattern (S1111111), so nothing needs remapping.
    """
    assert 0 <= exp_min <= exp_max <= 14, f"{name}: exponent band {exp_min}..{exp_max} out of range"
    w = words(name, count, offset)
    span = exp_max - exp_min + 1
    exponent = (np.uint8(exp_min) + ((w >> np.uint64(32)) % np.uint64(span)).astype(np.uint8)).astype(np.uint8)
    mantissa = ((w >> np.uint64(56)) & np.uint64(0x7)).astype(np.uint8)
    sign = ((w >> np.uint64(63)).astype(np.uint8) << np.uint8(7)) if signed else np.uint8(0)
    return (sign | (exponent << np.uint8(3)) | mantissa).astype(np.uint8)


def f4_nibbles(name: str, count: int, offset: int = 0) -> np.ndarray:
    """`count` f4e2m1 codes packed two per byte, low nibble first.

    Every one of the 16 patterns is a finite f4e2m1 value, so unlike `f8_codes` there is
    nothing to remap. `count` must be even. Nibbles are addressed *per element* so that
    element `i` depends on index `i`, not on the byte it happens to share.

    Note the `offset` parameter has **no counterpart** in `test_kernels.rs`'s `prng`
    module, which is otherwise a byte-for-byte mirror of this file. Nothing needs a
    chunked f4 tensor today; adding one here without adding `offset` there would silently
    produce two different tensors. Mirror it first.
    """
    assert count % 2 == 0, f"{name}: f4 element count must be even, got {count}"
    assert offset % 2 == 0, f"{name}: f4 offset must be even, got {offset}"
    codes = (words(name, count, offset) >> np.uint64(60)).astype(np.uint8) & np.uint8(0xF)
    return (codes[0::2] | (codes[1::2] << np.uint8(4))).astype(np.uint8)



def checksum(storage: np.ndarray, word_offset: int = 0) -> int:
    """A position-weighted 32-bit checksum over a tensor's storage bytes.

    Position-weighted so a permutation is caught and not just a value change, and
    expressible as a vectorized reduction here and a flat loop in Rust. Not a
    cryptographic hash -- its whole job is catching an implementation divergence between
    two files that are meant to be the same algorithm.

    `word_offset` is the tensor-global index of this block's first 32-bit word, so a
    tensor too large to hold at once can be checksummed a block at a time and the
    results summed (mod 2**32). The lm_head weight needs this: 1.007e9 elements is 2 GB.
    Blocks must therefore be a whole number of 32-bit words.

    The padding assert below has no counterpart in `test_kernels.rs`, which zero-pads
    unconditionally. Every checksummed tensor's byte length is a multiple of 4 today, so
    the two agree; a chunked tensor whose final block needed padding would be rejected
    here and silently accepted there.
    """
    data = np.ascontiguousarray(storage).view(np.uint8)
    padding = (-data.size) % 4
    if padding:
        assert word_offset == 0, "only the final block of a chunked tensor may need padding"
        data = np.concatenate([data, np.zeros(padding, dtype=np.uint8)])
    total = 0
    for start in range(0, data.size, CHUNK * 4):
        block = data[start : start + CHUNK * 4].view(np.uint32)
        index = np.arange(block.size, dtype=np.uint32) + np.uint32(word_offset + start // 4)
        total += int((block * (index * np.uint32(2) + np.uint32(1))).sum(dtype=np.uint32))
    return total & 0xFFFF_FFFF

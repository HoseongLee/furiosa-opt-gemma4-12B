//! Kernel fixture test for the 3 decoder-layer kernels in `src/ops.rs`
//! (`sliding_project_qkv`, `sliding_attention_output`, `decoder_feedforward`) -- the
//! source of truth for Stage 1 correctness and performance. A `--bin`, not a `[[test]]`:
//! `cargo furiosa-opt`'s kernel-registry discovery does not pick up `launch()` calls made
//! from an integration test crate.
//!
//! **Inputs are synthesized, not stored.** `ref/fixtures.safetensors` (written by
//! `scripts/generate_references.py`) carries only expected outputs and per-tensor
//! checksums; every input is synthesized here from the `prng` module below, which mirrors
//! `scripts/fixture_prng.py` byte-for-byte. `Synth::verify` checks the synthesized bytes
//! against the fixture's checksum before any kernel launches, so a divergence between the
//! two implementations fails immediately by tensor name instead of as a numeric error.
//!
//! **Each kernel is graded over `RUNS` independently-seeded configurations, not one.**
//! Every synthesized value -- activations, weights, scales, the RoPE position -- differs
//! per run, so a kernel can't special-case whatever the fixture always used to hand it
//! (`sliding_attention_output`'s `x` was once fixed at exactly `bf16(+1)`/`bf16(-1)`, which
//! let its consumer get away with a sign-magnitude shortcut). A kernel must pass every run
//! to count as correct; the score reported per kernel is the *median* cycle count across
//! the sweep, not any single run's number. Runs execute batched (all of one kernel's runs,
//! then the next) rather than interleaved -- measured to have equal-or-lower variance.

use std::collections::HashMap;
use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use furiosa_opt_std::prelude::*;

use furiosa_opt_gemma4::axes::*;
use furiosa_opt_gemma4::{Chip, ops};

mod prng {
    const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
    const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
    const MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;
    const MIX_B: u64 = 0x94D0_49BB_1331_11EB;

    pub fn name_hash(name: &str) -> u64 {
        let mut hash = FNV_OFFSET;
        for byte in name.as_bytes() {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME);
        }
        hash
    }

    pub fn word(seed: u64, index: usize) -> u64 {
        let mut x = (seed ^ index as u64).wrapping_add(GOLDEN);
        x = (x ^ (x >> 30)).wrapping_mul(MIX_A);
        x = (x ^ (x >> 27)).wrapping_mul(MIX_B);
        x ^ (x >> 31)
    }

    pub fn u01(w: u64) -> f32 {
        ((w >> 40) as u32) as f32 / 16_777_216.0
    }

    pub fn f32_to_bf16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        let rounding = 0x7FFF + ((bits >> 16) & 1);
        (bits.wrapping_add(rounding) >> 16) as u16
    }

    pub fn bf16_bits_to_f32(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    fn push_u16(out: &mut Vec<u8>, value: u16) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    pub fn bf16_uniform(name: &str, count: usize, lo: f32, hi: f32) -> Vec<u8> {
        bf16_uniform_range(name, 0, count, lo, hi)
    }

    pub fn f32_uniform(name: &str, count: usize, lo: f32, hi: f32) -> Vec<f32> {
        let seed = name_hash(name);
        (0..count)
            .map(|index| lo + (hi - lo) * u01(word(seed, index)))
            .collect()
    }

    pub fn bf16_uniform_range(name: &str, offset: usize, count: usize, lo: f32, hi: f32) -> Vec<u8> {
        let seed = name_hash(name);
        let mut out = Vec::with_capacity(count * 2);
        for index in offset..offset + count {
            push_u16(&mut out, f32_to_bf16_bits(lo + (hi - lo) * u01(word(seed, index))));
        }
        out
    }

    pub fn bf16_signs(name: &str, count: usize, scale: f32) -> Vec<u8> {
        let seed = name_hash(name);
        let magnitude = f32_to_bf16_bits(scale.abs());
        let mut out = Vec::with_capacity(count * 2);
        for index in 0..count {
            let negative = word(seed, index) >> 63 == 1;
            push_u16(&mut out, if negative { magnitude | 0x8000 } else { magnitude });
        }
        out
    }

    pub fn f8_banded(name: &str, count: usize, exp_min: u8, exp_max: u8, signed: bool) -> Vec<u8> {
        assert!(
            exp_min <= exp_max && exp_max <= 14,
            "{name}: exponent band out of range"
        );
        let seed = name_hash(name);
        let span = u64::from(exp_max - exp_min + 1);
        let mut out = Vec::with_capacity(count);
        for index in 0..count {
            let w = word(seed, index);
            let exponent = exp_min + ((w >> 32) % span) as u8;
            let mantissa = ((w >> 56) & 0x7) as u8;
            let sign = if signed { ((w >> 63) as u8) << 7 } else { 0 };
            out.push(sign | (exponent << 3) | mantissa);
        }
        out
    }

    pub fn f4_nibbles(name: &str, count: usize) -> Vec<u8> {
        assert!(count % 2 == 0, "{name}: f4 element count must be even");
        let seed = name_hash(name);
        let mut out = Vec::with_capacity(count / 2);
        for pair in 0..count / 2 {
            let low = ((word(seed, pair * 2) >> 60) & 0xF) as u8;
            let high = ((word(seed, pair * 2 + 1) >> 60) & 0xF) as u8;
            out.push(low | (high << 4));
        }
        out
    }

    pub fn checksum(bytes: &[u8], word_offset: usize) -> u32 {
        let mut total: u32 = 0;
        for (block, chunk) in bytes.chunks(4).enumerate() {
            let mut word = [0u8; 4];
            word[..chunk.len()].copy_from_slice(chunk);
            let index = (word_offset + block) as u32;
            total = total.wrapping_add(u32::from_le_bytes(word).wrapping_mul(index.wrapping_mul(2).wrapping_add(1)));
        }
        total
    }
}

const WEIGHT_EXP: (u8, u8) = (7, 14);
const LOCAL_SCALE_EXP: (u8, u8) = (8, 10);
const ROW_SCALE: (f32, f32) = (0.0, 1e-3);
const UNIT: (f32, f32) = (0.0, 1.0);
const RMS_WEIGHT: (f32, f32) = (0.75, 1.25);
const ACTIVATION: (f32, f32) = (-1.0, 1.0);
const GLOBAL_SCALE: (f32, f32) = (2048.0, 16384.0);

const LAYER_SCALAR: f32 = 0.375;

/// Independent draws baked into one fixture (`generate_references.py`'s `RUNS`). The
/// loop lives here rather than in a shell script so a single binary invocation -- one
/// local run, one remote job -- sweeps all of them; the printed score is the median
/// across the sweep, not any single run's number.
///
/// No warmup discard: a 10-sweep, 270-sample study (2026-09) found no warmup shape in the
/// per-run cycle counts -- noise was spread evenly across all run positions, and for
/// `sliding_project_qkv` the first 4 runs were measurably *quieter* than the rest. The
/// dominant noise source looked like it was between separate process invocations (shared
/// hardware, thermal, scheduling), not a per-process cold start, so discarding early runs
/// only cost sample size for no reduction in variance.
const RUNS: usize = 3;

/// The RoPE/cache-offset position for run `run`. Mirrors `generate_references.py`'s
/// `_run_pos` exactly; never 0, since an all-zero position makes RoPE the identity
/// rotation for every frequency -- a single fixed configuration a kernel could
/// special-case, which is the same shape of bug `x`'s old `+/-1` span was.
fn pos(run: usize) -> usize {
    let seed = prng::name_hash(&format!("run{run}.global.pos"));
    1 + (prng::word(seed, 0) % 2047) as usize
}

/// Run `run`'s NVFP4 global scale for `{up,gate,down}`, drawn from `GLOBAL_SCALE` rather
/// than pinned to the real checkpoint's numbers (mirrors `generate_references.py`).
fn global_scale(run: usize, name: &str) -> f32 {
    prng::f32_uniform(
        &format!("run{run}.global.{name}_global_scale"),
        1,
        GLOBAL_SCALE.0,
        GLOBAL_SCALE.1,
    )[0]
}

/// The middle value of `RUNS` (odd, so no averaging ambiguity), used for the score a
/// kernel is graded on instead of any single run's cycle count -- one lucky or unlucky
/// run shouldn't move the number that matters.
fn median(values: &[u64]) -> u64 {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}

struct Fixture {
    expected: HashMap<String, Vec<f32>>,
    checksums: HashMap<String, u32>,
}

const FIXTURE_PATHS: [&str; 2] = ["ref/fixtures.safetensors", "fixtures.safetensors"];

fn fixture_path() -> String {
    if let Ok(path) = std::env::var("GEMMA4_FIXTURE") {
        return path;
    }
    FIXTURE_PATHS
        .iter()
        .find(|path| std::path::Path::new(path).exists())
        .unwrap_or(&FIXTURE_PATHS[0])
        .to_string()
}

impl Fixture {
    fn load(path: &str) -> Self {
        let file = File::open(path)
            .unwrap_or_else(|e| panic!("{path}: {e} -- run `python3 scripts/generate_references.py` first"));
        let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap_or_else(|e| panic!("{path}: {e}"));
        let tensors = safetensors::SafeTensors::deserialize(&mmap)
            .unwrap_or_else(|e| panic!("{path}: not a safetensors file: {e}"));

        let mut expected = HashMap::new();
        let mut checksums = HashMap::new();
        for (name, view) in tensors.tensors() {
            match view.dtype() {
                safetensors::Dtype::F32 => {
                    let values = view
                        .data()
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    expected.insert(name, values);
                }
                safetensors::Dtype::I64 => {
                    let raw = i64::from_le_bytes(view.data()[..8].try_into().unwrap());
                    checksums.insert(name, raw as u32);
                }
                other => panic!("{name}: unexpected fixture dtype {other:?}"),
            }
        }
        Self { expected, checksums }
    }

    fn expect(&self, run: usize, test: &str, label: &str) -> &[f32] {
        let key = format!("run{run}.{test}.{label}");
        self.expected
            .get(&key)
            .unwrap_or_else(|| panic!("fixture has no `{key}` -- regenerate with scripts/generate_references.py"))
    }

    fn assert_every_expectation_is_tested(&self) {
        let orphans: Vec<&str> = self
            .expected
            .keys()
            .map(String::as_str)
            .filter(|key| {
                // Keys are `run{N}.{test}.{label}`, and `label` itself may contain a dot
                // (e.g. `expected.q`), so the test name is specifically the 2nd segment.
                let test = key.split('.').nth(1).unwrap_or(key);
                !TESTS.iter().any(|candidate| candidate.name == test)
            })
            .collect();
        assert!(
            orphans.is_empty(),
            "the fixture has expectations no test reads, so they are silently unchecked: {orphans:?}\n\
             add the matching `Test` row, `run_test` arm and shim, or drop the generator"
        );
    }

    fn checksum(&self, run: usize, test: &str, input: &str) -> u32 {
        let key = format!("run{run}.{test}.check.{input}");
        *self.checksums.get(&key).unwrap_or_else(|| {
            panic!("fixture has no checksum `{key}` -- regenerate with scripts/generate_references.py")
        })
    }
}

struct Synth<'a> {
    test: &'static str,
    run: usize,
    fixture: &'a Fixture,
}

impl<'a> Synth<'a> {
    fn new(test: &'static str, run: usize, fixture: &'a Fixture) -> Self {
        Self { test, run, fixture }
    }

    fn seed(&self, name: &str) -> String {
        format!("run{}.{}.{}", self.run, self.test, name)
    }

    fn verify(&self, name: &str, storage: &[u8]) {
        let expected = self.fixture.checksum(self.run, self.test, name);
        let actual = prng::checksum(storage, 0);
        assert_eq!(
            actual, expected,
            "{}.{name}: synthesized input does not match scripts/generate_references.py \
             (checksum {actual:#010x} vs {expected:#010x}) -- fixture_prng.py and \
             this file's prng module have diverged",
            self.test
        );
    }

    async fn upload<D: MaterializableScalar, E: M>(
        &self,
        device: &mut Device,
        name: &str,
        storage: Vec<u8>,
    ) -> HbmTensor<D, Chip, E> {
        self.verify(name, &storage);
        HostTensor::<D, E>::from_buf(storage)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap()
    }

    async fn bf16<E: M>(&self, device: &mut Device, name: &str, span: (f32, f32)) -> HbmTensor<bf16, Chip, E> {
        let storage = prng::bf16_uniform(&self.seed(name), E::SIZE, span.0, span.1);
        self.upload(device, name, storage).await
    }

    async fn f8<E: M>(
        &self,
        device: &mut Device,
        name: &str,
        band: (u8, u8),
        signed: bool,
    ) -> HbmTensor<f8e4m3, Chip, E> {
        let storage = prng::f8_banded(&self.seed(name), E::SIZE, band.0, band.1, signed);
        self.upload(device, name, storage).await
    }

    async fn f4<E: M>(&self, device: &mut Device, name: &str) -> HbmTensor<f4e2m1, Chip, E> {
        let storage = prng::f4_nibbles(&self.seed(name), E::SIZE);
        self.upload(device, name, storage).await
    }

    async fn constant_f32<E: M>(&self, device: &mut Device, name: &str, values: &[f32]) -> HbmTensor<f32, Chip, E> {
        let storage: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.upload(device, name, storage).await
    }

    async fn constant_bf16<E: M>(&self, device: &mut Device, name: &str, values: &[f32]) -> HbmTensor<bf16, Chip, E> {
        let storage: Vec<u8> = values
            .iter()
            .flat_map(|v| prng::f32_to_bf16_bits(*v).to_le_bytes())
            .collect();
        self.upload(device, name, storage).await
    }

    async fn constant_i32<E: M>(&self, device: &mut Device, name: &str, value: i32) -> HbmTensor<i32, Chip, E> {
        self.upload(device, name, value.to_le_bytes().to_vec()).await
    }
}

async fn exact_rmsnorm_input(device: &mut Device, s: &Synth<'_>) -> HbmTensor<bf16, Chip, m![H]> {
    let signs = prng::bf16_signs(&s.seed("x_signs"), H::SIZE, 1.0);
    s.verify("x_signs", &signs);
    let weights = prng::bf16_uniform(&s.seed("input_rms_weight"), H::SIZE, RMS_WEIGHT.0, RMS_WEIGHT.1);

    let storage: Vec<u8> = signs
        .chunks_exact(2)
        .zip(weights.chunks_exact(2))
        .flat_map(|(sign, weight)| {
            let sign = prng::bf16_bits_to_f32(u16::from_le_bytes(sign.try_into().unwrap()));
            let weight = prng::bf16_bits_to_f32(u16::from_le_bytes(weight.try_into().unwrap()));
            prng::f32_to_bf16_bits(sign / weight).to_le_bytes()
        })
        .collect();
    s.verify("x_exact", &storage);
    HostTensor::<bf16, m![H]>::from_buf(storage)
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}

async fn zeros<D: ScalarBytes + MaterializableScalar, E: M>(device: &mut Device) -> HbmTensor<D, Chip, E> {
    HostTensor::<D, E>::from_buf(vec![0u8; E::SIZE * D::BITS / 8])
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}

async fn read_bf16<E: M>(device: &mut Device, tensor: &HbmTensor<bf16, Chip, E>) -> Vec<f32> {
    let host: HostTensor<bf16, E> = tensor.to_host(&mut device.pdma).await.unwrap();
    host.into_vec().into_iter().map(bf16::to_f32).collect()
}

fn rope_tables(head_dim: usize, theta: f64, partial_rotary_factor: f64, pos: usize) -> (Vec<f32>, Vec<f32>) {
    let angles = (partial_rotary_factor * head_dim as f64 / 2.0).floor() as usize;
    let half = head_dim / 2;
    let mut inv_freq = vec![0.0f64; half];
    for (i, slot) in inv_freq.iter_mut().enumerate().take(angles) {
        *slot = 1.0 / theta.powf((2 * i) as f64 / head_dim as f64);
    }

    let mut cos = vec![0.0f32; head_dim];
    let mut sin = vec![0.0f32; head_dim];
    for i in 0..head_dim {
        let angle = pos as f64 * inv_freq[i % half];
        cos[i] = angle.cos() as f32;
        sin[i] = angle.sin() as f32;
    }
    (cos, sin)
}

fn negate_low_half(sin: &[f32]) -> Vec<f32> {
    let half = sin.len() / 2;
    sin.iter()
        .enumerate()
        .map(|(i, v)| if i < half { -v } else { *v })
        .collect()
}

async fn rope_table<D: AxisName>(
    device: &mut Device,
    s: &Synth<'_>,
    name: &str,
    values: &[f32],
    pos: usize,
) -> HbmTensor<bf16, Chip, m![E, D]> {
    let row: Vec<u8> = values
        .iter()
        .flat_map(|v| prng::f32_to_bf16_bits(*v).to_le_bytes())
        .collect();
    s.verify(name, &row);

    let mut storage = vec![0u8; E::SIZE * D::SIZE * 2];
    let byte_offset = pos * D::SIZE * 2;
    storage[byte_offset..byte_offset + row.len()].copy_from_slice(&row);
    HostTensor::<bf16, m![E, D]>::from_buf(storage)
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}

struct Test {
    name: &'static str,
    atol: f32,
    rtol: f32,
}

const RTOL: f32 = 1e-2;

const TESTS: &[Test] = &[
    Test {
        name: "sliding_project_qkv",
        atol: 0.04,
        rtol: RTOL,
    },
    Test {
        name: "sliding_attention_output",
        atol: 0.05,
        rtol: RTOL,
    },
    Test {
        name: "decoder_feedforward",
        atol: 0.01,
        rtol: RTOL,
    },
];

async fn run_test(
    device: &mut Device,
    fixture: &Fixture,
    name: &'static str,
    run: usize,
) -> Vec<(&'static str, Vec<f32>)> {
    match name {
        "sliding_project_qkv" => sliding_project_qkv(device, fixture, run).await,
        "sliding_attention_output" => sliding_attention_output(device, fixture, run).await,
        "decoder_feedforward" => decoder_feedforward(device, fixture, run).await,
        other => panic!("no shim for test `{other}` -- add one in run_test"),
    }
}

async fn sliding_project_qkv(device: &mut Device, fixture: &Fixture, run: usize) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new("sliding_project_qkv", run, fixture);
    let pos = pos(run);

    let input_rms_weight: HbmTensor<bf16, Chip, m![H]> = s.bf16(device, "input_rms_weight", RMS_WEIGHT).await;
    let x: HbmTensor<bf16, Chip, m![H]> = exact_rmsnorm_input(device, &s).await;

    let q_weight: HbmTensor<f8e4m3, Chip, m![Qs, H]> = s.f8(device, "q_weight", WEIGHT_EXP, true).await;
    let k_weight: HbmTensor<f8e4m3, Chip, m![Ps, H]> = s.f8(device, "k_weight", WEIGHT_EXP, true).await;
    let v_weight: HbmTensor<f8e4m3, Chip, m![Ps, H]> = s.f8(device, "v_weight", WEIGHT_EXP, true).await;
    let q_weight_scale: HbmTensor<bf16, Chip, m![Qs]> = s.bf16(device, "q_weight_scale", ROW_SCALE).await;
    let k_weight_scale: HbmTensor<bf16, Chip, m![Ps]> = s.bf16(device, "k_weight_scale", ROW_SCALE).await;
    let v_weight_scale: HbmTensor<bf16, Chip, m![Ps]> = s.bf16(device, "v_weight_scale", ROW_SCALE).await;
    let q_rms_weight: HbmTensor<bf16, Chip, m![Ds]> = s.bf16(device, "q_rms_weight", UNIT).await;
    let k_rms_weight: HbmTensor<bf16, Chip, m![Ds]> = s.bf16(device, "k_rms_weight", UNIT).await;

    let (cos_values, sin_values) = rope_tables(Ds::SIZE, 10_000.0, 1.0, pos);
    let cos: HbmTensor<bf16, Chip, m![E, Ds]> = rope_table::<Ds>(device, &s, "cos", &cos_values, pos).await;
    let sin: HbmTensor<bf16, Chip, m![E, Ds]> =
        rope_table::<Ds>(device, &s, "sin", &negate_low_half(&sin_values), pos).await;
    let rope_offset: HbmTensor<i32, Chip, m![1]> =
        s.constant_i32(device, "rope_offset", (pos * Ds::SIZE * 2) as i32).await;

    let slot = pos % Ts::SIZE;
    let offset = (slot * Ns::SIZE * Ds::SIZE * 2) as i32;
    let kv_offset: HbmTensor<i32, Chip, m![1]> = s.constant_i32(device, "kv_offset", offset).await;

    let mut k_cache: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]> = zeros(device).await;
    let mut v_cache: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]> = zeros(device).await;
    let mut q_out: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]> = zeros(device).await;

    launch(
        ops::sliding_project_qkv,
        (
            device,
            &x,
            &q_weight,
            &k_weight,
            &v_weight,
            &q_weight_scale,
            &k_weight_scale,
            &v_weight_scale,
            &input_rms_weight,
            &q_rms_weight,
            &k_rms_weight,
            &kv_offset,
            &rope_offset,
            &cos,
            &sin,
            &mut k_cache,
            &mut v_cache,
            &mut q_out,
        ),
    )
    .await
    .unwrap();

    let width = Ns::SIZE * Ds::SIZE;
    let k = read_bf16(device, &k_cache).await[slot * width..(slot + 1) * width].to_vec();
    let v = read_bf16(device, &v_cache).await[slot * width..(slot + 1) * width].to_vec();
    vec![
        ("expected.q", read_bf16(device, &q_out).await),
        ("expected.k", k),
        ("expected.v", v),
    ]
}

async fn sliding_attention_output(device: &mut Device, fixture: &Fixture, run: usize) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new("sliding_attention_output", run, fixture);
    let x: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]> = s.bf16(device, "x", ACTIVATION).await;
    let post_attn_rms_weight: HbmTensor<bf16, Chip, m![H]> = s.bf16(device, "post_attn_rms_weight", UNIT).await;
    let o_weight: HbmTensor<f8e4m3, Chip, m![H, Qs]> = s.f8(device, "o_weight", WEIGHT_EXP, true).await;
    let o_weight_scale: HbmTensor<bf16, Chip, m![H]> = s.bf16(device, "o_weight_scale", ROW_SCALE).await;
    let mut residual: HbmTensor<bf16, Chip, m![H]> = s.bf16(device, "residual", UNIT).await;

    launch(
        ops::sliding_attention_output,
        (
            device,
            &x,
            &post_attn_rms_weight,
            &o_weight,
            &o_weight_scale,
            &mut residual,
        ),
    )
    .await
    .unwrap();
    vec![("expected", read_bf16(device, &residual).await)]
}

async fn decoder_feedforward(device: &mut Device, fixture: &Fixture, run: usize) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new("decoder_feedforward", run, fixture);

    let mut residual: HbmTensor<bf16, Chip, m![H]> = s.bf16(device, "residual", UNIT).await;
    let pre_ff_rms_weight: HbmTensor<bf16, Chip, m![H]> = s.bf16(device, "pre_ff_rms_weight", UNIT).await;
    let post_ff_rms_weight: HbmTensor<bf16, Chip, m![H]> = s.bf16(device, "post_ff_rms_weight", UNIT).await;

    let up_weight_packed: HbmTensor<f4e2m1, Chip, m![L, H]> = s.f4(device, "up_weight_packed").await;
    let gate_weight_packed: HbmTensor<f4e2m1, Chip, m![L, H]> = s.f4(device, "gate_weight_packed").await;
    let down_weight_packed: HbmTensor<f4e2m1, Chip, m![H, L]> = s.f4(device, "down_weight_packed").await;
    let up_weight_scale: HbmTensor<f8e4m3, Chip, m![L, H / 16]> =
        s.f8(device, "up_weight_scale", LOCAL_SCALE_EXP, false).await;
    let gate_weight_scale: HbmTensor<f8e4m3, Chip, m![L, H / 16]> =
        s.f8(device, "gate_weight_scale", LOCAL_SCALE_EXP, false).await;
    let down_weight_scale: HbmTensor<f8e4m3, Chip, m![H, L / 16]> =
        s.f8(device, "down_weight_scale", LOCAL_SCALE_EXP, false).await;

    let up_global_scale: HbmTensor<f32, Chip, m![1]> = s
        .constant_f32(device, "up_global_scale", &[1.0 / global_scale(run, "up")])
        .await;
    let gate_global_scale: HbmTensor<f32, Chip, m![1]> = s
        .constant_f32(device, "gate_global_scale", &[1.0 / global_scale(run, "gate")])
        .await;
    let down_global_scale: HbmTensor<f32, Chip, m![1]> = s
        .constant_f32(device, "down_global_scale", &[1.0 / global_scale(run, "down")])
        .await;

    let layer_scalar: HbmTensor<bf16, Chip, m![1 # 8]> =
        s.constant_bf16(device, "layer_scalar", &[LAYER_SCALAR; 8]).await;

    launch(
        ops::decoder_feedforward,
        (
            device,
            &mut residual,
            &pre_ff_rms_weight,
            &up_weight_packed,
            &gate_weight_packed,
            &down_weight_packed,
            &up_weight_scale,
            &gate_weight_scale,
            &down_weight_scale,
            &up_global_scale,
            &gate_global_scale,
            &down_global_scale,
            &post_ff_rms_weight,
            &layer_scalar,
        ),
    )
    .await
    .unwrap();
    vec![("expected", read_bf16(device, &residual).await)]
}

fn compare(label: &str, expected: &[f32], actual: &[f32], atol: f32, rtol: f32) -> bool {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{label}: shape mismatch ({} expected vs {} from device)",
        expected.len(),
        actual.len()
    );

    let mut max_diff = 0.0f32;
    let mut max_index = 0usize;
    let mut sum_diff = 0.0f64;
    let mut within = 0usize;

    for (index, (&want, &got)) in expected.iter().zip(actual).enumerate() {
        if !got.is_finite() {
            println!("[{label:34}] FAIL -- non-finite device output at {index}");
            return false;
        }
        let diff = (want - got).abs();
        if diff <= atol + rtol * want.abs() {
            within += 1;
        }
        if diff > max_diff {
            max_diff = diff;
            max_index = index;
        }
        sum_diff += f64::from(diff);
    }

    let count = expected.len();
    let ok = within == count;
    let relative = if expected[max_index].abs() > 1e-12 {
        max_diff / expected[max_index].abs() * 100.0
    } else {
        0.0
    };
    println!(
        "[{label:34}] max|Δ|={max_diff:9.5} ({relative:7.2}% of expected)  mean|Δ|={:9.6}  \
         within tol={:6.2}%  -> {}",
        sum_diff / count as f64,
        within as f32 / count as f32 * 100.0,
        if ok { "PASS" } else { "FAIL" }
    );
    ok
}

// --- on-device cycle collection (only armed when FURIOSA_OPT_PROFILE is set) ---

const TRACING_TARGET_NPU: &str = "span::npu";

#[derive(Clone, Copy)]
struct Span {
    cluster: u64,
    begin: u64,
    end: u64,
}

/// A minimal `tracing::Subscriber`: all we need is to see each `span::npu` span's fields
/// as it's created, not the full `Layer`/`Registry` machinery `tracing-subscriber` offers.
#[derive(Clone, Default)]
struct Collector {
    spans: Arc<Mutex<Vec<Span>>>,
    next_id: Arc<AtomicU64>,
}

impl Collector {
    fn clear(&self) {
        self.spans.lock().unwrap().clear();
    }

    /// Real cycles for whatever ran since the last `clear`: each cluster counts its own
    /// cycles (see `Function::run`'s `tid = span.cluster`), so a span's `begin`/`end` is
    /// only comparable to another span on the *same* cluster -- taking a raw min/max across
    /// clusters mixes unrelated counters and produces a meaningless span. This takes the
    /// union of spans within each cluster, then the longest cluster's window.
    fn window_cycles(&self) -> Option<u64> {
        let spans = self.spans.lock().unwrap();
        let mut per_cluster: HashMap<u64, (u64, u64)> = HashMap::new();
        for span in spans.iter() {
            let window = per_cluster.entry(span.cluster).or_insert((span.begin, span.end));
            window.0 = window.0.min(span.begin);
            window.1 = window.1.max(span.end);
        }
        per_cluster
            .values()
            .map(|(begin, end)| end.saturating_sub(*begin))
            .max()
    }
}

#[derive(Default)]
struct FieldExtractor {
    begin: Option<u64>,
    end: Option<u64>,
    cluster: Option<u64>,
}

impl tracing::field::Visit for FieldExtractor {
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        match field.name() {
            "begin_cycle" => self.begin = Some(value),
            "end_cycle" => self.end = Some(value),
            "tid" => self.cluster = Some(value),
            _ => {}
        }
    }

    fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}
}

impl tracing::Subscriber for Collector {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.target() == TRACING_TARGET_NPU
    }

    fn new_span(&self, attrs: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        if attrs.metadata().target() == TRACING_TARGET_NPU {
            let mut extractor = FieldExtractor::default();
            attrs.record(&mut extractor);
            if let (Some(cluster), Some(begin), Some(end)) = (extractor.cluster, extractor.begin, extractor.end) {
                self.spans.lock().unwrap().push(Span { cluster, begin, end });
            }
        }
        // 0 is reserved by `span::Id`; spans aren't tracked individually here, so the
        // id only needs to be unique and non-zero.
        tracing::span::Id::from_u64(self.next_id.fetch_add(1, Ordering::Relaxed) + 1)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
    fn event(&self, _event: &tracing::Event<'_>) {}
    fn enter(&self, _span: &tracing::span::Id) {}
    fn exit(&self, _span: &tracing::span::Id) {}
}

fn profiling_enabled() -> bool {
    let level = std::env::var("FURIOSA_OPT_PROFILE")
        .unwrap_or_default()
        .to_ascii_lowercase();
    matches!(level.as_str(), "info" | "debug" | "trace")
}

fn settle() -> Duration {
    let ms = std::env::var("GEMMA4_PROFILE_SETTLE_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500u64);
    Duration::from_millis(ms)
}

#[tokio::main]
async fn main() {
    let fixture = Fixture::load(&fixture_path());
    fixture.assert_every_expectation_is_tested();
    let mut device = Device::new(ops::sliding_project_qkv.topology()).unwrap();

    let profile = profiling_enabled();
    let collector = Collector::default();
    if profile {
        tracing::subscriber::set_global_default(collector.clone()).expect("set global tracing subscriber");
    }
    let settle = settle();

    println!(
        "NPU kernel tests -- {} cases x {RUNS} independently-seeded runs against a precomputed reference{}\n",
        TESTS.len(),
        if profile { ", with on-device cycle counts" } else { "" }
    );

    // Batched (all runs of one kernel, then all runs of the next), not interleaved: a
    // 20-sweep, 420-sample study (2026-09) found interleaving run0 of A/B/C, then run1 of
    // A/B/C, ... measurably increased variance for 2 of 3 kernels (and tied on the third)
    // versus running each kernel's whole sweep back to back. Switching between different
    // kernels' code and access patterns every run looks like it adds noise (icache/branch
    // predictor churn) rather than removing any bias from slow drift over a sweep.
    let mut failures = Vec::new();
    for test in TESTS {
        if profile {
            println!("==> {}", test.name);
        }

        let mut all_ok = true;
        let mut cycles_by_run = Vec::with_capacity(RUNS);
        for run in 0..RUNS {
            if profile {
                collector.clear();
            }

            let outputs = run_test(&mut device, &fixture, test.name, run).await;

            let cycles = if profile {
                // Spans are decoded off the launch hot path during deferred read-back,
                // not synchronously with `run_test(..).await` returning.
                tokio::time::sleep(settle).await;
                collector.window_cycles()
            } else {
                None
            };

            assert!(
                !outputs.is_empty(),
                "{}: shim produced no outputs to compare",
                test.name
            );
            let mut ok = true;
            for (label, actual) in &outputs {
                let base = if outputs.len() == 1 {
                    test.name.to_string()
                } else {
                    format!("{} {}", test.name, label.trim_start_matches("expected."))
                };
                let display = format!("{base} run{run}");
                ok &= compare(
                    &display,
                    fixture.expect(run, test.name, label),
                    actual,
                    test.atol,
                    test.rtol,
                );
            }
            all_ok &= ok;
            if let Some(c) = cycles {
                cycles_by_run.push(c);
            }
        }

        if profile {
            if cycles_by_run.is_empty() {
                println!("    cycles=none observed");
            } else {
                println!(
                    "    median cycles={} (of {} runs: {:?})",
                    median(&cycles_by_run),
                    cycles_by_run.len(),
                    cycles_by_run
                );
            }
            println!();
        }

        if !all_ok {
            failures.push(test.name);
        }
    }

    println!();
    if failures.is_empty() {
        println!("all {} tests passed", TESTS.len());
    } else {
        println!(
            "{} of {} tests failed: {}",
            failures.len(),
            TESTS.len(),
            failures.join(", ")
        );
    }
    std::process::exit(i32::from(!failures.is_empty()));
}


use std::collections::HashMap;
use std::fs::File;

use furiosa_opt_std::prelude::*;

use furiosa_opt_gemma4::axes::*;
use furiosa_opt_gemma4::{Chip, ops, ops_audio, ops_vision};

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

    pub fn f32_uniform(name: &str, count: usize, lo: f32, hi: f32) -> Vec<u8> {
        let seed = name_hash(name);
        let mut out = Vec::with_capacity(count * 4);
        for index in 0..count {
            let value = lo + (hi - lo) * u01(word(seed, index));
            out.extend_from_slice(&value.to_le_bytes());
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
const CENTRED: (f32, f32) = (-0.5, 0.5);
const RMS_WEIGHT: (f32, f32) = (0.75, 1.25);

const POS: usize = 137;
const SLIDING_VALID: usize = 300;
const FULL_ROW: usize = 137;
const LAYER_SCALAR: f32 = 0.375;
const EMBED_TOKEN_ID: usize = 90_210;

const RAW_GLOBAL_SCALES: [f32; 3] = [9600.0, 9600.0, 12928.0];

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

    fn expect(&self, test: &str, label: &str) -> &[f32] {
        let key = format!("{test}.{label}");
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
                let test = key.split('.').next().unwrap_or(key);
                !TESTS.iter().any(|candidate| candidate.name == test)
            })
            .collect();
        assert!(
            orphans.is_empty(),
            "the fixture has expectations no test reads, so they are silently unchecked: {orphans:?}\n\
             add the matching `Test` row, `run_test` arm and shim, or drop the generator"
        );
    }

    fn checksum(&self, test: &str, input: &str) -> u32 {
        let key = format!("{test}.check.{input}");
        *self.checksums.get(&key).unwrap_or_else(|| {
            panic!("fixture has no checksum `{key}` -- regenerate with scripts/generate_references.py")
        })
    }
}

struct Synth<'a> {
    test: &'static str,
    fixture: &'a Fixture,
}

impl<'a> Synth<'a> {
    fn new(test: &'static str, fixture: &'a Fixture) -> Self {
        Self { test, fixture }
    }

    fn seed(&self, name: &str) -> String {
        format!("{}.{}", self.test, name)
    }

    fn verify(&self, name: &str, storage: &[u8]) {
        let expected = self.fixture.checksum(self.test, name);
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
        ctx: &mut Context,
        name: &str,
        storage: Vec<u8>,
    ) -> HbmTensor<D, Chip, E> {
        self.verify(name, &storage);
        HostTensor::<D, E>::from_buf(storage).to_hbm(&mut ctx.pdma).await
    }

    async fn bf16<E: M>(&self, ctx: &mut Context, name: &str, span: (f32, f32)) -> HbmTensor<bf16, Chip, E> {
        let storage = prng::bf16_uniform(&self.seed(name), E::SIZE, span.0, span.1);
        self.upload(ctx, name, storage).await
    }

    async fn signs<E: M>(&self, ctx: &mut Context, name: &str, scale: f32) -> HbmTensor<bf16, Chip, E> {
        let storage = prng::bf16_signs(&self.seed(name), E::SIZE, scale);
        self.upload(ctx, name, storage).await
    }

    async fn f8<E: M>(
        &self,
        ctx: &mut Context,
        name: &str,
        band: (u8, u8),
        signed: bool,
    ) -> HbmTensor<f8e4m3, Chip, E> {
        let storage = prng::f8_banded(&self.seed(name), E::SIZE, band.0, band.1, signed);
        self.upload(ctx, name, storage).await
    }

    async fn f4<E: M>(&self, ctx: &mut Context, name: &str) -> HbmTensor<f4e2m1, Chip, E> {
        let storage = prng::f4_nibbles(&self.seed(name), E::SIZE);
        self.upload(ctx, name, storage).await
    }

    async fn f32<E: M>(&self, ctx: &mut Context, name: &str, span: (f32, f32)) -> HbmTensor<f32, Chip, E> {
        let storage = prng::f32_uniform(&self.seed(name), E::SIZE, span.0, span.1);
        self.upload(ctx, name, storage).await
    }

    async fn constant_f32<E: M>(&self, ctx: &mut Context, name: &str, values: &[f32]) -> HbmTensor<f32, Chip, E> {
        let storage: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.upload(ctx, name, storage).await
    }

    async fn constant_bf16<E: M>(&self, ctx: &mut Context, name: &str, values: &[f32]) -> HbmTensor<bf16, Chip, E> {
        let storage: Vec<u8> = values
            .iter()
            .flat_map(|v| prng::f32_to_bf16_bits(*v).to_le_bytes())
            .collect();
        self.upload(ctx, name, storage).await
    }

    async fn constant_i32<E: M>(&self, ctx: &mut Context, name: &str, value: i32) -> HbmTensor<i32, Chip, E> {
        self.upload(ctx, name, value.to_le_bytes().to_vec()).await
    }
}

async fn exact_rmsnorm_input(ctx: &mut Context, s: &Synth<'_>) -> HbmTensor<bf16, Chip, m![H]> {
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
    HostTensor::<bf16, m![H]>::from_buf(storage).to_hbm(&mut ctx.pdma).await
}

async fn zeros<D: ScalarBytes + MaterializableScalar, E: M>(ctx: &mut Context) -> HbmTensor<D, Chip, E> {
    HostTensor::<D, E>::from_buf(vec![0u8; E::SIZE * D::BITS / 8])
        .to_hbm(&mut ctx.pdma)
        .await
}

async fn read_bf16<E: M>(ctx: &mut Context, tensor: &HbmTensor<bf16, Chip, E>) -> Vec<f32> {
    let host: HostTensor<bf16, E> = tensor.to_host(&mut ctx.pdma).await;
    host.into_vec().into_iter().map(bf16::to_f32).collect()
}

async fn read_f32<E: M>(ctx: &mut Context, tensor: &HbmTensor<f32, Chip, E>) -> Vec<f32> {
    let host: HostTensor<f32, E> = tensor.to_host(&mut ctx.pdma).await;
    host.into_vec()
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

fn prefix_mask(len: usize, valid: usize) -> Vec<f32> {
    (0..len).map(|i| if i < valid { 1.0 } else { 0.0 }).collect()
}

async fn rope_table<D: AxisName>(
    ctx: &mut Context,
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
        .to_hbm(&mut ctx.pdma)
        .await
}

struct Test {
    name: &'static str,
    atol: f32,
    rtol: f32,
}

const RTOL: f32 = 2e-2;

const TESTS: &[Test] = &[
    Test {
        name: "copy_hidden_state",
        atol: 0.0,
        rtol: 0.0,
    },
    Test {
        name: "embed_token",
        atol: 0.25,
        rtol: RTOL,
    },
    Test {
        name: "sliding_project_qkv_exact",
        atol: 0.0625,
        rtol: RTOL,
    },
    Test {
        name: "sliding_project_qkv_random",
        atol: 0.0625,
        rtol: RTOL,
    },
    Test {
        name: "sliding_attention",
        atol: 2e-3,
        rtol: RTOL,
    },
    Test {
        name: "sliding_attention_output_exact",
        atol: 0.1,
        rtol: RTOL,
    },
    Test {
        name: "sliding_attention_output_random",
        atol: 0.1,
        rtol: RTOL,
    },
    Test {
        name: "full_project_qkv_exact",
        atol: 0.0625,
        rtol: RTOL,
    },
    Test {
        name: "full_project_qkv_random",
        atol: 0.0625,
        rtol: RTOL,
    },
    Test {
        name: "full_attention",
        atol: 2e-3,
        rtol: RTOL,
    },
    Test {
        name: "full_attention_output_exact",
        atol: 0.0625,
        rtol: RTOL,
    },
    Test {
        name: "full_attention_output_random",
        atol: 0.0625,
        rtol: RTOL,
    },
    Test {
        name: "decoder_feedforward",
        atol: 0.02,
        rtol: RTOL,
    },
    Test {
        name: "final_norm_and_logits",
        atol: 0.125,
        rtol: RTOL,
    },
    Test {
        name: "patch_embed",
        atol: 0.05,
        rtol: RTOL,
    },
    Test {
        name: "add_position_and_norm",
        atol: 0.02,
        rtol: RTOL,
    },
    Test {
        name: "project_to_text_embedding",
        atol: 0.5,
        rtol: RTOL,
    },
    Test {
        name: "audio_project_frame",
        atol: 1e-4,
        rtol: RTOL,
    },
    Test {
        name: "layernorm",
        atol: 0.02,
        rtol: RTOL,
    },
];

async fn run_test(ctx: &mut Context, fixture: &Fixture, name: &'static str) -> Vec<(&'static str, Vec<f32>)> {
    match name {
        "embed_token" => embed_token(ctx, fixture).await,
        "copy_hidden_state" => copy_hidden_state(ctx, fixture).await,
        "sliding_project_qkv_exact" => sliding_project_qkv(ctx, fixture, name, true).await,
        "sliding_project_qkv_random" => sliding_project_qkv(ctx, fixture, name, false).await,
        "sliding_attention" => sliding_attention(ctx, fixture).await,
        "sliding_attention_output_exact" => sliding_attention_output(ctx, fixture, name, true).await,
        "sliding_attention_output_random" => sliding_attention_output(ctx, fixture, name, false).await,
        "full_project_qkv_exact" => full_project_qkv(ctx, fixture, name, true).await,
        "full_project_qkv_random" => full_project_qkv(ctx, fixture, name, false).await,
        "full_attention" => full_attention(ctx, fixture).await,
        "full_attention_output_exact" => full_attention_output(ctx, fixture, name, true).await,
        "full_attention_output_random" => full_attention_output(ctx, fixture, name, false).await,
        "decoder_feedforward" => decoder_feedforward(ctx, fixture).await,
        "final_norm_and_logits" => final_norm_and_logits(ctx, fixture).await,
        "patch_embed" => patch_embed(ctx, fixture).await,
        "add_position_and_norm" => add_position_and_norm(ctx, fixture).await,
        "project_to_text_embedding" => project_to_text_embedding(ctx, fixture).await,
        "audio_project_frame" => audio_project_frame(ctx, fixture).await,
        "layernorm" => layernorm(ctx, fixture).await,
        other => panic!("no shim for test `{other}` -- add one in run_test"),
    }
}

async fn embed_token(ctx: &mut Context, fixture: &Fixture) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new("embed_token", fixture);

    let row = prng::bf16_uniform(&s.seed("row"), H::SIZE, CENTRED.0, CENTRED.1);
    s.verify("row", &row);
    let byte_offset = EMBED_TOKEN_ID * H::SIZE * 2;
    let mut storage = vec![0u8; W::SIZE * H::SIZE * 2];
    storage[byte_offset..byte_offset + row.len()].copy_from_slice(&row);
    let embedding_table: HbmTensor<bf16, Chip, m![W, H]> = HostTensor::<bf16, m![W, H]>::from_buf(storage)
        .to_hbm(&mut ctx.pdma)
        .await;

    let offset: HbmTensor<i32, Chip, m![1]> = s.constant_i32(ctx, "offset", byte_offset as i32).await;
    let mut out: HbmTensor<bf16, Chip, m![H]> = zeros(ctx).await;

    launch(ops::embed_token, (ctx, &embedding_table, &offset, &mut out)).await;
    vec![("expected", read_bf16(ctx, &out).await)]
}

async fn copy_hidden_state(ctx: &mut Context, fixture: &Fixture) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new("copy_hidden_state", fixture);
    let x: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "x", CENTRED).await;
    let mut out: HbmTensor<bf16, Chip, m![H]> = zeros(ctx).await;

    launch(ops::copy_hidden_state, (ctx, &x, &mut out)).await;
    vec![("expected", read_bf16(ctx, &out).await)]
}

async fn sliding_project_qkv(
    ctx: &mut Context,
    fixture: &Fixture,
    name: &'static str,
    exact: bool,
) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new(name, fixture);

    let input_rms_weight: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "input_rms_weight", RMS_WEIGHT).await;
    let x: HbmTensor<bf16, Chip, m![H]> = if exact {
        exact_rmsnorm_input(ctx, &s).await
    } else {
        s.signs(ctx, "x_signs", 1.0).await
    };

    let q_weight: HbmTensor<f8e4m3, Chip, m![Qs, H]> = s.f8(ctx, "q_weight", WEIGHT_EXP, true).await;
    let k_weight: HbmTensor<f8e4m3, Chip, m![Ps, H]> = s.f8(ctx, "k_weight", WEIGHT_EXP, true).await;
    let v_weight: HbmTensor<f8e4m3, Chip, m![Ps, H]> = s.f8(ctx, "v_weight", WEIGHT_EXP, true).await;
    let q_weight_scale: HbmTensor<bf16, Chip, m![Qs]> = s.bf16(ctx, "q_weight_scale", ROW_SCALE).await;
    let k_weight_scale: HbmTensor<bf16, Chip, m![Ps]> = s.bf16(ctx, "k_weight_scale", ROW_SCALE).await;
    let v_weight_scale: HbmTensor<bf16, Chip, m![Ps]> = s.bf16(ctx, "v_weight_scale", ROW_SCALE).await;
    let q_rms_weight: HbmTensor<bf16, Chip, m![Ds]> = s.bf16(ctx, "q_rms_weight", UNIT).await;
    let k_rms_weight: HbmTensor<bf16, Chip, m![Ds]> = s.bf16(ctx, "k_rms_weight", UNIT).await;

    let (cos_values, sin_values) = rope_tables(Ds::SIZE, 10_000.0, 1.0, POS);
    let cos: HbmTensor<bf16, Chip, m![E, Ds]> = rope_table::<Ds>(ctx, &s, "cos", &cos_values, POS).await;
    let sin: HbmTensor<bf16, Chip, m![E, Ds]> =
        rope_table::<Ds>(ctx, &s, "sin", &negate_low_half(&sin_values), POS).await;
    let rope_offset: HbmTensor<i32, Chip, m![1]> =
        s.constant_i32(ctx, "rope_offset", (POS * Ds::SIZE * 2) as i32).await;

    let slot = POS % Ts::SIZE;
    let offset = (slot * Ns::SIZE * Ds::SIZE * 2) as i32;
    let kv_offset: HbmTensor<i32, Chip, m![1]> = s.constant_i32(ctx, "kv_offset", offset).await;

    let mut k_cache: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]> = zeros(ctx).await;
    let mut v_cache: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]> = zeros(ctx).await;
    let mut q_out: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]> = zeros(ctx).await;

    launch(
        ops::sliding_project_qkv,
        (
            ctx,
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
    .await;

    let width = Ns::SIZE * Ds::SIZE;
    let k = read_bf16(ctx, &k_cache).await[slot * width..(slot + 1) * width].to_vec();
    let v = read_bf16(ctx, &v_cache).await[slot * width..(slot + 1) * width].to_vec();
    vec![
        ("expected.q", read_bf16(ctx, &q_out).await),
        ("expected.k", k),
        ("expected.v", v),
    ]
}

async fn sliding_attention(ctx: &mut Context, fixture: &Fixture) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new("sliding_attention", fixture);
    let q: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]> = s.signs(ctx, "q", 1.0 / 16.0).await;
    let k: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]> = s.signs(ctx, "k", 1.0).await;
    let v: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]> = s.bf16(ctx, "v", CENTRED).await;
    let mask: HbmTensor<f32, Chip, m![Ts]> = s.constant_f32(ctx, "mask", &prefix_mask(Ts::SIZE, SLIDING_VALID)).await;
    let mut out: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]> = zeros(ctx).await;

    launch(ops::sliding_attention, (ctx, &q, &k, &v, &mask, &mut out)).await;
    vec![("expected", read_bf16(ctx, &out).await)]
}

async fn sliding_attention_output(
    ctx: &mut Context,
    fixture: &Fixture,
    name: &'static str,
    exact: bool,
) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new(name, fixture);
    let x: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]> = if exact {
        s.signs(ctx, "x", 1.0).await
    } else {
        s.bf16(ctx, "x", CENTRED).await
    };
    let post_attn_rms_weight: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "post_attn_rms_weight", UNIT).await;
    let o_weight: HbmTensor<f8e4m3, Chip, m![H, Qs]> = s.f8(ctx, "o_weight", WEIGHT_EXP, true).await;
    let o_weight_scale: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "o_weight_scale", ROW_SCALE).await;
    let mut residual: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "residual", UNIT).await;

    launch(
        ops::sliding_attention_output,
        (
            ctx,
            &x,
            &post_attn_rms_weight,
            &o_weight,
            &o_weight_scale,
            &mut residual,
        ),
    )
    .await;
    vec![("expected", read_bf16(ctx, &residual).await)]
}

async fn full_project_qkv(
    ctx: &mut Context,
    fixture: &Fixture,
    name: &'static str,
    exact: bool,
) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new(name, fixture);

    let input_rms_weight: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "input_rms_weight", RMS_WEIGHT).await;
    let x: HbmTensor<bf16, Chip, m![H]> = if exact {
        exact_rmsnorm_input(ctx, &s).await
    } else {
        s.signs(ctx, "x_signs", 1.0).await
    };

    let q_weight: HbmTensor<f8e4m3, Chip, m![Qf, H]> = s.f8(ctx, "q_weight", WEIGHT_EXP, true).await;
    let k_weight: HbmTensor<f8e4m3, Chip, m![Pf, H]> = s.f8(ctx, "k_weight", WEIGHT_EXP, true).await;
    let q_weight_scale: HbmTensor<bf16, Chip, m![Qf]> = s.bf16(ctx, "q_weight_scale", ROW_SCALE).await;
    let k_weight_scale: HbmTensor<bf16, Chip, m![Pf]> = s.bf16(ctx, "k_weight_scale", ROW_SCALE).await;
    let q_rms_weight: HbmTensor<bf16, Chip, m![Df]> = s.bf16(ctx, "q_rms_weight", UNIT).await;
    let k_rms_weight: HbmTensor<bf16, Chip, m![Df]> = s.bf16(ctx, "k_rms_weight", UNIT).await;

    let (cos_values, sin_values) = rope_tables(Df::SIZE, 1_000_000.0, 0.25, POS);
    let cos: HbmTensor<bf16, Chip, m![E, Df]> = rope_table::<Df>(ctx, &s, "cos", &cos_values, POS).await;
    let sin: HbmTensor<bf16, Chip, m![E, Df]> =
        rope_table::<Df>(ctx, &s, "sin", &negate_low_half(&sin_values), POS).await;
    let rope_offset: HbmTensor<i32, Chip, m![1]> =
        s.constant_i32(ctx, "rope_offset", (POS * Df::SIZE * 2) as i32).await;
    let kv_offset: HbmTensor<i32, Chip, m![1]> = s.constant_i32(ctx, "kv_offset", (POS * Df::SIZE * 2) as i32).await;

    let mut k_cache: HbmTensor<bf16, Chip, m![Tf, Df]> = zeros(ctx).await;
    let mut v_cache: HbmTensor<bf16, Chip, m![Tf, Df]> = zeros(ctx).await;
    let mut q_out: HbmTensor<bf16, Chip, m![Gf, Df]> = zeros(ctx).await;

    launch(
        ops::full_project_qkv,
        (
            ctx,
            &x,
            &q_weight,
            &k_weight,
            &q_weight_scale,
            &k_weight_scale,
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
    .await;

    let k = read_bf16(ctx, &k_cache).await[POS * Df::SIZE..(POS + 1) * Df::SIZE].to_vec();
    let v = read_bf16(ctx, &v_cache).await[POS * Df::SIZE..(POS + 1) * Df::SIZE].to_vec();
    vec![
        ("expected.q", read_bf16(ctx, &q_out).await),
        ("expected.k", k),
        ("expected.v", v),
    ]
}

async fn full_attention(ctx: &mut Context, fixture: &Fixture) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new("full_attention", fixture);
    let q: HbmTensor<bf16, Chip, m![Gf, Df]> = s.signs(ctx, "q", 1.0 / 16.0).await;
    let k0: HbmTensor<bf16, Chip, m![Tf, Df]> = s.signs(ctx, "k0", 1.0).await;
    let k1: HbmTensor<bf16, Chip, m![Tf, Df]> = s.signs(ctx, "k1", 1.0).await;
    let v0: HbmTensor<bf16, Chip, m![Tf, Df]> = s.bf16(ctx, "v0", CENTRED).await;
    let v1: HbmTensor<bf16, Chip, m![Tf, Df]> = s.bf16(ctx, "v1", CENTRED).await;
    let mask0: HbmTensor<f32, Chip, m![Tf]> = s.constant_f32(ctx, "mask0", &prefix_mask(Tf::SIZE, Tf::SIZE)).await;
    let mask1: HbmTensor<f32, Chip, m![Tf]> = s.constant_f32(ctx, "mask1", &prefix_mask(Tf::SIZE, FULL_ROW + 1)).await;

    let mut running_max: HbmTensor<f32, Chip, m![Gf]> = zeros(ctx).await;
    let mut running_sum: HbmTensor<f32, Chip, m![Gf]> = zeros(ctx).await;
    let mut out: HbmTensor<bf16, Chip, m![Gf, Df]> = zeros(ctx).await;

    launch(
        ops::full_attention_first_page,
        (ctx, &q, &k0, &v0, &mask0, &mut running_max, &mut running_sum, &mut out),
    )
    .await;
    launch(
        ops::full_attention_page,
        (ctx, &q, &k1, &v1, &mask1, &mut running_max, &mut running_sum, &mut out),
    )
    .await;

    let sums = read_f32(ctx, &running_sum).await;
    let normalized = read_bf16(ctx, &out)
        .await
        .into_iter()
        .enumerate()
        .map(|(i, value)| value / sums[i / Df::SIZE])
        .collect();
    vec![("expected", normalized)]
}

async fn full_attention_output(
    ctx: &mut Context,
    fixture: &Fixture,
    name: &'static str,
    exact: bool,
) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new(name, fixture);
    let x: HbmTensor<bf16, Chip, m![Gf, Df]> = if exact {
        s.signs(ctx, "x", 1.0).await
    } else {
        s.bf16(ctx, "x", CENTRED).await
    };
    let running_sum: HbmTensor<f32, Chip, m![Gf]> = if exact {
        s.constant_f32(ctx, "running_sum", &vec![4.0f32; Gf::SIZE]).await
    } else {
        s.f32(ctx, "running_sum", (2.0, 4.0)).await
    };
    let post_attn_rms_weight: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "post_attn_rms_weight", UNIT).await;
    let o_weight: HbmTensor<f8e4m3, Chip, m![H, Qf]> = s.f8(ctx, "o_weight", WEIGHT_EXP, true).await;
    let o_weight_scale: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "o_weight_scale", ROW_SCALE).await;
    let mut residual: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "residual", UNIT).await;

    launch(
        ops::full_attention_output,
        (
            ctx,
            &x,
            &running_sum,
            &post_attn_rms_weight,
            &o_weight,
            &o_weight_scale,
            &mut residual,
        ),
    )
    .await;
    vec![("expected", read_bf16(ctx, &residual).await)]
}

async fn decoder_feedforward(ctx: &mut Context, fixture: &Fixture) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new("decoder_feedforward", fixture);

    let mut residual: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "residual", UNIT).await;
    let pre_ff_rms_weight: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "pre_ff_rms_weight", UNIT).await;
    let post_ff_rms_weight: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "post_ff_rms_weight", UNIT).await;

    let up_weight_packed: HbmTensor<f4e2m1, Chip, m![L, H]> = s.f4(ctx, "up_weight_packed").await;
    let gate_weight_packed: HbmTensor<f4e2m1, Chip, m![L, H]> = s.f4(ctx, "gate_weight_packed").await;
    let down_weight_packed: HbmTensor<f4e2m1, Chip, m![H, L]> = s.f4(ctx, "down_weight_packed").await;
    let up_weight_scale: HbmTensor<f8e4m3, Chip, m![L, H / 16]> =
        s.f8(ctx, "up_weight_scale", LOCAL_SCALE_EXP, false).await;
    let gate_weight_scale: HbmTensor<f8e4m3, Chip, m![L, H / 16]> =
        s.f8(ctx, "gate_weight_scale", LOCAL_SCALE_EXP, false).await;
    let down_weight_scale: HbmTensor<f8e4m3, Chip, m![H, L / 16]> =
        s.f8(ctx, "down_weight_scale", LOCAL_SCALE_EXP, false).await;

    let up_global_scale: HbmTensor<f32, Chip, m![1]> = s
        .constant_f32(ctx, "up_global_scale", &[1.0 / RAW_GLOBAL_SCALES[0]])
        .await;
    let gate_global_scale: HbmTensor<f32, Chip, m![1]> = s
        .constant_f32(ctx, "gate_global_scale", &[1.0 / RAW_GLOBAL_SCALES[1]])
        .await;
    let down_global_scale: HbmTensor<f32, Chip, m![1]> = s
        .constant_f32(ctx, "down_global_scale", &[1.0 / RAW_GLOBAL_SCALES[2]])
        .await;

    let layer_scalar: HbmTensor<bf16, Chip, m![1 # 8]> = s.constant_bf16(ctx, "layer_scalar", &[LAYER_SCALAR; 8]).await;

    launch(
        ops::decoder_feedforward,
        (
            ctx,
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
    .await;
    vec![("expected", read_bf16(ctx, &residual).await)]
}

async fn final_norm_and_logits(ctx: &mut Context, fixture: &Fixture) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new("final_norm_and_logits", fixture);
    let x: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "x", UNIT).await;
    let rms_weight: HbmTensor<bf16, Chip, m![H]> = s.bf16(ctx, "rms_weight", UNIT).await;

    let seed = s.seed("lm_head_weight");
    let rows_per_block = (1 << 23) / H::SIZE;
    let mut storage = Vec::with_capacity(W::SIZE * H::SIZE * 2);
    let mut running: u32 = 0;
    let mut start = 0;
    while start < W::SIZE {
        let rows = rows_per_block.min(W::SIZE - start);
        let block = prng::bf16_uniform_range(&seed, start * H::SIZE, rows * H::SIZE, CENTRED.0, CENTRED.1);
        running = running.wrapping_add(prng::checksum(&block, start * H::SIZE / 2));
        storage.extend_from_slice(&block);
        start += rows;
    }
    let recorded = fixture.checksum("final_norm_and_logits", "lm_head_weight");
    assert_eq!(
        running, recorded,
        "final_norm_and_logits.lm_head_weight: synthesized input does not match \
         scripts/generate_references.py ({running:#010x} vs {recorded:#010x})"
    );
    let lm_head_weight: HbmTensor<bf16, Chip, m![W, H]> = HostTensor::<bf16, m![W, H]>::from_buf(storage)
        .to_hbm(&mut ctx.pdma)
        .await;

    let mut out: HbmTensor<bf16, Chip, m![W]> = zeros(ctx).await;
    launch(
        ops::final_norm_and_logits,
        (ctx, &x, &rms_weight, &lm_head_weight, &mut out),
    )
    .await;
    vec![("expected", read_bf16(ctx, &out).await)]
}

async fn patch_embed(ctx: &mut Context, fixture: &Fixture) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new("patch_embed", fixture);
    let x: HbmTensor<bf16, Chip, m![Rv]> = s.bf16(ctx, "x", CENTRED).await;
    let ln1_weight: HbmTensor<bf16, Chip, m![Rv]> = s.bf16(ctx, "ln1_weight", UNIT).await;
    let ln1_bias: HbmTensor<bf16, Chip, m![Rv]> = s.bf16(ctx, "ln1_bias", CENTRED).await;
    let dense_weight: HbmTensor<bf16, Chip, m![Mv, Rv]> = s.bf16(ctx, "dense_weight", CENTRED).await;
    let dense_bias: HbmTensor<bf16, Chip, m![Mv]> = s.bf16(ctx, "dense_bias", CENTRED).await;
    let ln2_weight: HbmTensor<bf16, Chip, m![Mv]> = s.bf16(ctx, "ln2_weight", UNIT).await;
    let ln2_bias: HbmTensor<bf16, Chip, m![Mv]> = s.bf16(ctx, "ln2_bias", CENTRED).await;
    let mut out: HbmTensor<bf16, Chip, m![Mv]> = zeros(ctx).await;

    launch(
        ops_vision::patch_embed,
        (
            ctx,
            &x,
            &ln1_weight,
            &ln1_bias,
            &dense_weight,
            &dense_bias,
            &ln2_weight,
            &ln2_bias,
            &mut out,
        ),
    )
    .await;
    vec![("expected", read_bf16(ctx, &out).await)]
}

async fn add_position_and_norm(ctx: &mut Context, fixture: &Fixture) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new("add_position_and_norm", fixture);
    let x: HbmTensor<bf16, Chip, m![Mv]> = s.bf16(ctx, "x", CENTRED).await;
    let pos_embed: HbmTensor<bf16, Chip, m![Mv]> = s.bf16(ctx, "pos_embed", CENTRED).await;
    let ln3_weight: HbmTensor<bf16, Chip, m![Mv]> = s.bf16(ctx, "ln3_weight", UNIT).await;
    let ln3_bias: HbmTensor<bf16, Chip, m![Mv]> = s.bf16(ctx, "ln3_bias", CENTRED).await;
    let mut out: HbmTensor<bf16, Chip, m![Mv]> = zeros(ctx).await;

    launch(
        ops_vision::add_position_and_norm,
        (ctx, &x, &pos_embed, &ln3_weight, &ln3_bias, &mut out),
    )
    .await;
    vec![("expected", read_bf16(ctx, &out).await)]
}

async fn project_to_text_embedding(ctx: &mut Context, fixture: &Fixture) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new("project_to_text_embedding", fixture);
    let x: HbmTensor<bf16, Chip, m![Ov]> = s.bf16(ctx, "x", CENTRED).await;
    let proj_weight: HbmTensor<bf16, Chip, m![H, Ov]> = s.bf16(ctx, "proj_weight", CENTRED).await;
    let mut out: HbmTensor<bf16, Chip, m![H]> = zeros(ctx).await;

    launch(ops_vision::project_to_text_embedding, (ctx, &x, &proj_weight, &mut out)).await;
    vec![("expected", read_bf16(ctx, &out).await)]
}

async fn audio_project_frame(ctx: &mut Context, fixture: &Fixture) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new("audio_project_frame", fixture);
    let x: HbmTensor<bf16, Chip, m![Aa]> = s.bf16(ctx, "x", CENTRED).await;
    let proj_weight: HbmTensor<bf16, Chip, m![H, Aa]> = s.bf16(ctx, "proj_weight", CENTRED).await;
    let mut out: HbmTensor<bf16, Chip, m![H]> = zeros(ctx).await;

    launch(ops_audio::audio_project_frame, (ctx, &x, &proj_weight, &mut out)).await;
    vec![("expected", read_bf16(ctx, &out).await)]
}

async fn layernorm(ctx: &mut Context, fixture: &Fixture) -> Vec<(&'static str, Vec<f32>)> {
    let s = Synth::new("layernorm", fixture);
    let x: HbmTensor<bf16, Chip, m![Mv]> = s.bf16(ctx, "x", CENTRED).await;
    let weight: HbmTensor<bf16, Chip, m![Mv]> = s.bf16(ctx, "weight", UNIT).await;
    let bias: HbmTensor<bf16, Chip, m![Mv]> = s.bf16(ctx, "bias", CENTRED).await;
    let mut out: HbmTensor<bf16, Chip, m![Mv]> = zeros(ctx).await;

    launch(ops_vision::layernorm, (ctx, &x, &weight, &bias, &mut out)).await;
    vec![("expected", read_bf16(ctx, &out).await)]
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

#[tokio::main]
async fn main() {
    let fixture = Fixture::load(&fixture_path());
    fixture.assert_every_expectation_is_tested();
    let mut ctx = Context::acquire();

    println!(
        "NPU kernel tests -- {} cases against a precomputed reference\n",
        TESTS.len()
    );

    let mut failures = Vec::new();
    for test in TESTS {
        let outputs = run_test(&mut ctx, &fixture, test.name).await;
        assert!(
            !outputs.is_empty(),
            "{}: shim produced no outputs to compare",
            test.name
        );
        let mut ok = true;
        for (label, actual) in &outputs {
            let display = if outputs.len() == 1 {
                test.name.to_string()
            } else {
                format!("{} {}", test.name, label.trim_start_matches("expected."))
            };
            ok &= compare(&display, fixture.expect(test.name, label), actual, test.atol, test.rtol);
        }
        if !ok {
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

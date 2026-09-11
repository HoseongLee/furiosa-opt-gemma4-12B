use furiosa_opt_std::prelude::*;

use crate::axes::{Aa, C, Df, Ds, E, Gf, Gs, H, Mv, Ns, Pv, Rv, Tf, Ts, W};
use crate::host::load::{FullLayer, Layer, Model, SlidingLayer};
use crate::{Chip, ops, ops_audio, ops_vision};

fn gather_pos_embedding(model: &Model, col: usize, row: usize) -> Vec<bf16> {
    let mv = Mv::SIZE;
    let col = col.min(Pv::SIZE - 1);
    let row = row.min(Pv::SIZE - 1);
    let col_row = &model.vision.pos_embedding[col * 2 * mv..col * 2 * mv + mv];
    let row_row = &model.vision.pos_embedding[(row * 2 + 1) * mv..(row * 2 + 1) * mv + mv];
    col_row
        .iter()
        .zip(row_row)
        .map(|(&a, &b)| bf16::from_f32(a.to_f32() + b.to_f32()))
        .collect()
}

pub async fn encode_vision_patch(
    device: &mut Device,
    model: &Model,
    pixels: &[bf16],
    position: (usize, usize),
) -> Result<HbmTensor<bf16, Chip, m![H]>, Error> {
    assert_eq!(pixels.len(), Rv::SIZE, "vision patch must be Rv-wide");
    let x: HbmTensor<bf16, Chip, m![Rv]> = HostTensor::<bf16, m![Rv]>::from_vec(pixels.to_vec())
        .to_hbm(&mut device.pdma)
        .await?;

    let mut patch_out: HbmTensor<bf16, Chip, m![Mv]> =
        HostTensor::<bf16, m![Mv]>::zero().to_hbm(&mut device.pdma).await?;
    launch(
        ops_vision::patch_embed,
        (
            device,
            &x,
            &model.vision.patch_ln1_weight,
            &model.vision.patch_ln1_bias,
            &model.vision.patch_dense_weight,
            &model.vision.patch_dense_bias,
            &model.vision.patch_ln2_weight,
            &model.vision.patch_ln2_bias,
            &mut patch_out,
        ),
    )
    .await?;

    let (col, row) = position;
    let pos_embed: HbmTensor<bf16, Chip, m![Mv]> =
        HostTensor::<bf16, m![Mv]>::from_vec(gather_pos_embedding(model, col, row))
            .to_hbm(&mut device.pdma)
            .await?;

    let mut normed: HbmTensor<bf16, Chip, m![Mv]> = HostTensor::<bf16, m![Mv]>::zero().to_hbm(&mut device.pdma).await?;
    launch(
        ops_vision::add_position_and_norm,
        (
            device,
            &patch_out,
            &pos_embed,
            &model.vision.pos_norm_weight,
            &model.vision.pos_norm_bias,
            &mut normed,
        ),
    )
    .await?;

    let mut projected: HbmTensor<bf16, Chip, m![H]> =
        HostTensor::<bf16, m![H]>::zero().to_hbm(&mut device.pdma).await?;
    launch(
        ops_vision::project_to_text_embedding,
        (
            device,
            &unsafe { normed.reshape() },
            &model.vision.embedding_projection_weight,
            &mut projected,
        ),
    )
    .await?;

    Ok(projected)
}

pub async fn encode_audio_frame(
    device: &mut Device,
    model: &Model,
    samples: &[bf16],
) -> Result<HbmTensor<bf16, Chip, m![H]>, Error> {
    assert_eq!(samples.len(), Aa::SIZE, "audio frame must be Aa-wide");
    let input: HbmTensor<bf16, Chip, m![Aa]> = HostTensor::<bf16, m![Aa]>::from_vec(samples.to_vec())
        .to_hbm(&mut device.pdma)
        .await?;
    let mut output: HbmTensor<bf16, Chip, m![H]> = HostTensor::<bf16, m![H]>::zero().to_hbm(&mut device.pdma).await?;
    launch(
        ops_audio::audio_project_frame,
        (device, &input, &model.audio.embedding_projection_weight, &mut output),
    )
    .await?;
    Ok(output)
}

enum KvCache {
    Sliding {
        k: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
        v: HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    },
    Full {
        k: Vec<HbmTensor<bf16, Chip, m![Tf, Df]>>,
        v: Vec<HbmTensor<bf16, Chip, m![Tf, Df]>>,
    },
}

impl KvCache {
    fn as_sliding_mut(
        &mut self,
    ) -> (
        &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
        &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    ) {
        match self {
            KvCache::Sliding { k, v } => (k, v),
            KvCache::Full { .. } => panic!("sliding layer must have a sliding KV cache"),
        }
    }

    fn as_full_mut(
        &mut self,
    ) -> (
        &mut Vec<HbmTensor<bf16, Chip, m![Tf, Df]>>,
        &mut Vec<HbmTensor<bf16, Chip, m![Tf, Df]>>,
    ) {
        match self {
            KvCache::Full { k, v } => (k, v),
            KvCache::Sliding { .. } => panic!("full layer must have a full KV cache"),
        }
    }
}

pub struct Workspace {
    cos_s: HbmTensor<bf16, Chip, m![E, Ds]>,
    sin_s: HbmTensor<bf16, Chip, m![E, Ds]>,
    cos_f: HbmTensor<bf16, Chip, m![E, Df]>,
    sin_f: HbmTensor<bf16, Chip, m![E, Df]>,
    mask_s: Vec<HbmTensor<f32, Chip, m![Ts]>>,
    mask_f_line: HbmTensor<f32, Chip, m![Tf]>,
    mask_f_diag: Vec<HbmTensor<f32, Chip, m![Tf]>>,
    kv_cache: Vec<KvCache>,
    offset_kv_s: HbmTensor<i32, Chip, m![1]>,
    offset_kv_f: HbmTensor<i32, Chip, m![1]>,
    offset_rope_s: HbmTensor<i32, Chip, m![1]>,
    offset_rope_f: HbmTensor<i32, Chip, m![1]>,
    pub logits: HbmTensor<bf16, Chip, m![W]>,
}

impl Workspace {
    pub async fn new(device: &mut Device, model: &Model) -> Result<Self, Error> {
        let (cos_s, sin_s) = upload_rope(device, Ds::SIZE, 10_000.0, Ds::SIZE).await?;
        let (cos_f, sin_f) = upload_rope(device, Df::SIZE, 1_000_000.0, 128).await?;
        let mask_s = upload_mask_rows::<m![Ts], _>(device, Ts::SIZE, |r, c| if c <= r { 1.0 } else { 0.0 }).await?;
        let mask_f_line = HostTensor::<f32, m![Tf]>::from_vec(vec![1.0; Tf::SIZE])
            .to_hbm(&mut device.pdma)
            .await?;
        let mask_f_diag =
            upload_mask_rows::<m![Tf], _>(device, Tf::SIZE, |r, c| if c <= r { 1.0 } else { 0.0 }).await?;

        let mut kv_cache = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            let cache = match layer {
                Layer::Sliding(_) => KvCache::Sliding {
                    k: HostTensor::<bf16, m![Ts, Ns, Ds]>::zero()
                        .to_hbm(&mut device.pdma)
                        .await?,
                    v: HostTensor::<bf16, m![Ts, Ns, Ds]>::zero()
                        .to_hbm(&mut device.pdma)
                        .await?,
                },
                Layer::Full(_) => KvCache::Full {
                    k: zero_hbm_vec::<bf16, m![Tf, Df]>(device, C::SIZE).await?,
                    v: zero_hbm_vec::<bf16, m![Tf, Df]>(device, C::SIZE).await?,
                },
            };
            kv_cache.push(cache);
        }

        let logits = HostTensor::<bf16, m![W]>::zero().to_hbm(&mut device.pdma).await?;

        let offset_kv_s = zero_offset(device).await?;
        let offset_kv_f = zero_offset(device).await?;
        let offset_rope_s = zero_offset(device).await?;
        let offset_rope_f = zero_offset(device).await?;

        Ok(Self {
            cos_s,
            sin_s,
            cos_f,
            sin_f,
            mask_s,
            mask_f_line,
            mask_f_diag,
            kv_cache,
            offset_kv_s,
            offset_kv_f,
            offset_rope_s,
            offset_rope_f,
            logits,
        })
    }

    /// A kernel writes each field wholesale before it is ever read, but the NPU backend still
    /// needs a real device allocation behind each `&mut` output argument at launch time, so
    /// every field is placed with a throwaway zero write rather than left as `HbmTensor::new()`
    /// (unplaced, for a kernel that returns its output instead of writing a parameter).
    pub async fn begin_decode(&mut self, device: &mut Device) -> Result<Decode, Error> {
        Ok(Decode {
            x: HostTensor::<bf16, m![H]>::zero().to_hbm(&mut device.pdma).await?,
            q_s: HostTensor::<bf16, m![Ns, Gs, Ds]>::zero()
                .to_hbm(&mut device.pdma)
                .await?,
            q_f: HostTensor::<bf16, m![Gf, Df]>::zero().to_hbm(&mut device.pdma).await?,
            attn_max: HostTensor::<f32, m![Gf]>::zero().to_hbm(&mut device.pdma).await?,
            attn_sum: HostTensor::<f32, m![Gf]>::zero().to_hbm(&mut device.pdma).await?,
            attn_s: HostTensor::<bf16, m![Ns, Gs, Ds]>::zero()
                .to_hbm(&mut device.pdma)
                .await?,
            attn_f: HostTensor::<bf16, m![Gf, Df]>::zero().to_hbm(&mut device.pdma).await?,
        })
    }
}

async fn set_position_offsets(device: &mut Device, ws: &mut Workspace, pos: usize) -> Result<(), Error> {
    let ps = pos % Ts::SIZE;
    let pf = pos % Tf::SIZE;

    let offset_kv_s: HostTensor<i32, m![1]> = HostTensor::from_vec([(ps * Ns::SIZE * Ds::SIZE * 2) as i32]);
    let offset_kv_f: HostTensor<i32, m![1]> = HostTensor::from_vec([(pf * Df::SIZE * 2) as i32]);

    let offset_rope_s: HostTensor<i32, m![1]> = HostTensor::from_vec([(pos * Ds::SIZE * 2) as i32]);
    let offset_rope_f: HostTensor<i32, m![1]> = HostTensor::from_vec([(pos * Df::SIZE * 2) as i32]);

    ws.offset_kv_s = offset_kv_s.to_hbm(&mut device.pdma).await?;
    ws.offset_kv_f = offset_kv_f.to_hbm(&mut device.pdma).await?;

    ws.offset_rope_s = offset_rope_s.to_hbm(&mut device.pdma).await?;
    ws.offset_rope_f = offset_rope_f.to_hbm(&mut device.pdma).await?;
    Ok(())
}

async fn zero_offset(device: &mut Device) -> Result<HbmTensor<i32, Chip, m![1]>, Error> {
    let zero: HostTensor<i32, m![1]> = HostTensor::from_vec([0]);
    zero.to_hbm(&mut device.pdma).await
}

async fn zero_hbm_vec<D: MaterializableScalar + num_traits::Zero, E: M>(
    device: &mut Device,
    count: usize,
) -> Result<Vec<HbmTensor<D, Chip, E>>, Error> {
    let mut tensors = Vec::with_capacity(count);
    for _ in 0..count {
        tensors.push(HostTensor::<D, E>::zero().to_hbm(&mut device.pdma).await?);
    }
    Ok(tensors)
}

async fn upload_mask_rows<Elt: M, F>(
    device: &mut Device,
    width: usize,
    mut value: F,
) -> Result<Vec<HbmTensor<f32, Chip, Elt>>, Error>
where
    F: FnMut(usize, usize) -> f32,
{
    let mut rows = Vec::with_capacity(width);
    for r in 0..width {
        let row: Vec<f32> = (0..width).map(|c| value(r, c)).collect();
        rows.push(HostTensor::<f32, Elt>::from_vec(row).to_hbm(&mut device.pdma).await?);
    }
    Ok(rows)
}

async fn upload_rope<D: AxisName>(
    device: &mut Device,
    dim: usize,
    theta: f32,
    rotated_dim: usize,
) -> Result<(HbmTensor<bf16, Chip, m![E, D]>, HbmTensor<bf16, Chip, m![E, D]>), Error> {
    let (cos, sin) = rope_data(dim, theta, rotated_dim);
    Ok((
        HostTensor::<bf16, m![E, D]>::from_vec(cos)
            .to_hbm(&mut device.pdma)
            .await?,
        HostTensor::<bf16, m![E, D]>::from_vec(sin)
            .to_hbm(&mut device.pdma)
            .await?,
    ))
}

fn rope_data(dim: usize, theta: f32, rotated_dim: usize) -> (Vec<bf16>, Vec<bf16>) {
    let half = dim / 2;
    let rotated_half = rotated_dim / 2;
    let mut cos = Vec::with_capacity(E::SIZE * dim);
    let mut sin = Vec::with_capacity(E::SIZE * dim);
    for pos in 0..E::SIZE {
        for i in 0..half {
            if i < rotated_half {
                let freq = (pos as f32) / theta.powf((2 * i) as f32 / dim as f32);
                cos.push(bf16::from_f32(freq.cos()));
                sin.push(bf16::from_f32(-freq.sin()));
            } else {
                cos.push(bf16::from_f32(1.0));
                sin.push(bf16::from_f32(0.0));
            }
        }
        for i in 0..half {
            if i < rotated_half {
                let freq = (pos as f32) / theta.powf((2 * i) as f32 / dim as f32);
                cos.push(bf16::from_f32(freq.cos()));
                sin.push(bf16::from_f32(freq.sin()));
            } else {
                cos.push(bf16::from_f32(1.0));
                sin.push(bf16::from_f32(0.0));
            }
        }
    }
    (cos, sin)
}

pub struct Decode {
    x: HbmTensor<bf16, Chip, m![H]>,
    q_s: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    q_f: HbmTensor<bf16, Chip, m![Gf, Df]>,
    attn_max: HbmTensor<f32, Chip, m![Gf]>,
    attn_sum: HbmTensor<f32, Chip, m![Gf]>,
    attn_s: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    attn_f: HbmTensor<bf16, Chip, m![Gf, Df]>,
}

impl Decode {
    pub async fn run(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        token: usize,
        model: &Model,
        pos: usize,
        compute_logits: bool,
    ) -> Result<(), Error> {
        let offset: HostTensor<i32, m![1]> = HostTensor::from_vec([(token * H::SIZE * 2) as i32]);
        let offset: HbmTensor<i32, Chip, m![1]> = offset.to_hbm(&mut device.pdma).await?;

        launch(ops::embed_token, (device, &model.embedding_table, &offset, &mut self.x)).await?;
        self.run_layers(device, ws, model, pos, compute_logits).await
    }

    pub async fn run_with_embedding(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        embedding: HbmTensor<bf16, Chip, m![H]>,
        model: &Model,
        pos: usize,
        compute_logits: bool,
    ) -> Result<(), Error> {
        self.x = embedding;
        self.run_layers(device, ws, model, pos, compute_logits).await
    }

    async fn run_layers(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        model: &Model,
        pos: usize,
        compute_logits: bool,
    ) -> Result<(), Error> {
        assert!(pos < E::SIZE, "position exceeds runtime context");
        set_position_offsets(device, ws, pos).await?;
        for (layer_index, layer) in model.layers.iter().enumerate() {
            match layer {
                Layer::Sliding(layer) => self.sliding(device, ws, layer_index, layer, pos).await?,
                Layer::Full(layer) => self.full(device, ws, layer_index, layer, pos).await?,
            }
        }

        if compute_logits {
            launch(
                ops::final_norm_and_logits,
                (
                    device,
                    &self.x,
                    &model.final_norm,
                    &model.embedding_table,
                    &mut ws.logits,
                ),
            )
            .await?;
        }
        Ok(())
    }

    async fn sliding(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        layer_index: usize,
        layer: &SlidingLayer,
        pos: usize,
    ) -> Result<(), Error> {
        let mask_row = pos.min(Ts::SIZE - 1);
        let (k, v) = ws.kv_cache[layer_index].as_sliding_mut();
        launch(
            ops::sliding_project_qkv,
            (
                device,
                &self.x,
                &layer.q_weight,
                &layer.k_weight,
                &layer.v_weight,
                &layer.q_weight_scale,
                &layer.k_weight_scale,
                &layer.v_weight_scale,
                &layer.input_norm,
                &layer.q_norm,
                &layer.k_norm,
                &ws.offset_kv_s,
                &ws.offset_rope_s,
                &ws.cos_s,
                &ws.sin_s,
                &mut *k,
                &mut *v,
                &mut self.q_s,
            ),
        )
        .await?;
        launch(
            ops::sliding_attention,
            (device, &self.q_s, &*k, &*v, &ws.mask_s[mask_row], &mut self.attn_s),
        )
        .await?;
        launch(
            ops::sliding_attention_output,
            (
                device,
                &self.attn_s,
                &layer.post_attention_norm,
                &layer.o_weight,
                &layer.o_weight_scale,
                &mut self.x,
            ),
        )
        .await?;
        self.mlp(
            device,
            &layer.mlp,
            &layer.pre_feedforward_norm,
            &layer.post_feedforward_norm,
            &layer.layer_scalar,
        )
        .await
    }

    async fn full(
        &mut self,
        device: &mut Device,
        ws: &mut Workspace,
        layer_index: usize,
        layer: &FullLayer,
        pos: usize,
    ) -> Result<(), Error> {
        let page = pos / Tf::SIZE;
        let row = pos % Tf::SIZE;
        assert!(page < C::SIZE, "position exceeds full-attention cache");
        let (k_cache, v_cache) = ws.kv_cache[layer_index].as_full_mut();
        launch(
            ops::full_project_qkv,
            (
                device,
                &self.x,
                &layer.q_weight,
                &layer.k_weight,
                &layer.q_weight_scale,
                &layer.k_weight_scale,
                &layer.input_norm,
                &layer.q_norm,
                &layer.k_norm,
                &ws.offset_kv_f,
                &ws.offset_rope_f,
                &ws.cos_f,
                &ws.sin_f,
                &mut k_cache[page],
                &mut v_cache[page],
                &mut self.q_f,
            ),
        )
        .await?;

        for kv_page in 0..=page {
            let mask = if kv_page == page {
                &ws.mask_f_diag[row]
            } else {
                &ws.mask_f_line
            };
            let args = (
                &mut *device,
                &self.q_f,
                &k_cache[kv_page],
                &v_cache[kv_page],
                mask,
                &mut self.attn_max,
                &mut self.attn_sum,
                &mut self.attn_f,
            );
            if kv_page == 0 {
                launch(ops::full_attention_first_page, args).await?;
            } else {
                launch(ops::full_attention_page, args).await?;
            }
        }

        launch(
            ops::full_attention_output,
            (
                device,
                &self.attn_f,
                &self.attn_sum,
                &layer.post_attention_norm,
                &layer.o_weight,
                &layer.o_weight_scale,
                &mut self.x,
            ),
        )
        .await?;
        self.mlp(
            device,
            &layer.mlp,
            &layer.pre_feedforward_norm,
            &layer.post_feedforward_norm,
            &layer.layer_scalar,
        )
        .await
    }

    async fn mlp(
        &mut self,
        device: &mut Device,
        mlp: &crate::host::load::MlpWeights,
        pre_feedforward_norm: &HbmTensor<bf16, Chip, m![H]>,
        post_feedforward_norm: &HbmTensor<bf16, Chip, m![H]>,
        layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
    ) -> Result<(), Error> {
        launch(
            ops::decoder_feedforward,
            (
                device,
                &mut self.x,
                pre_feedforward_norm,
                &mlp.up_weight_packed,
                &mlp.gate_weight_packed,
                &mlp.down_weight_packed,
                &mlp.up_weight_scale,
                &mlp.gate_weight_scale,
                &mlp.down_weight_scale,
                &mlp.up_global_scale,
                &mlp.gate_global_scale,
                &mlp.down_global_scale,
                post_feedforward_norm,
                layer_scalar,
            ),
        )
        .await
    }
}

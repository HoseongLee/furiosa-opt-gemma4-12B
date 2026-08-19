
use std::fs::File;
use std::path::{Path, PathBuf};

use furiosa_opt_std::prelude::*;

use crate::axes::{Aa, Df, Ds, Dummy2, H, L, Mv, Ov, Pf, Ps, Pv, Qf, Qs, Rv, W};
use crate::{Chip, LAYERS};

pub(crate) struct MlpWeights {
    pub up_weight_packed: HbmTensor<f4e2m1, Chip, m![L, H]>,
    pub gate_weight_packed: HbmTensor<f4e2m1, Chip, m![L, H]>,
    pub down_weight_packed: HbmTensor<f4e2m1, Chip, m![H, L]>,
    pub up_weight_scale: HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    pub gate_weight_scale: HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    pub down_weight_scale: HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    pub up_global_scale: HbmTensor<f32, Chip, m![1]>,
    pub gate_global_scale: HbmTensor<f32, Chip, m![1]>,
    pub down_global_scale: HbmTensor<f32, Chip, m![1]>,
}

pub(crate) struct SlidingLayer {
    pub input_norm: HbmTensor<bf16, Chip, m![H]>,
    pub q_weight: HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    pub k_weight: HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    pub v_weight: HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    pub q_weight_scale: HbmTensor<bf16, Chip, m![Qs]>,
    pub k_weight_scale: HbmTensor<bf16, Chip, m![Ps]>,
    pub v_weight_scale: HbmTensor<bf16, Chip, m![Ps]>,
    pub q_norm: HbmTensor<bf16, Chip, m![Ds]>,
    pub k_norm: HbmTensor<bf16, Chip, m![Ds]>,
    pub post_attention_norm: HbmTensor<bf16, Chip, m![H]>,
    pub o_weight: HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    pub o_weight_scale: HbmTensor<bf16, Chip, m![H]>,
    pub pre_feedforward_norm: HbmTensor<bf16, Chip, m![H]>,
    pub post_feedforward_norm: HbmTensor<bf16, Chip, m![H]>,
    pub layer_scalar: HbmTensor<bf16, Chip, m![1 # 8]>,
    pub mlp: MlpWeights,
}

pub(crate) struct FullLayer {
    pub input_norm: HbmTensor<bf16, Chip, m![H]>,
    pub q_weight: HbmTensor<f8e4m3, Chip, m![Qf, H]>,
    pub k_weight: HbmTensor<f8e4m3, Chip, m![Pf, H]>,
    pub q_weight_scale: HbmTensor<bf16, Chip, m![Qf]>,
    pub k_weight_scale: HbmTensor<bf16, Chip, m![Pf]>,
    pub q_norm: HbmTensor<bf16, Chip, m![Df]>,
    pub k_norm: HbmTensor<bf16, Chip, m![Df]>,
    pub post_attention_norm: HbmTensor<bf16, Chip, m![H]>,
    pub o_weight: HbmTensor<f8e4m3, Chip, m![H, Qf]>,
    pub o_weight_scale: HbmTensor<bf16, Chip, m![H]>,
    pub pre_feedforward_norm: HbmTensor<bf16, Chip, m![H]>,
    pub post_feedforward_norm: HbmTensor<bf16, Chip, m![H]>,
    pub layer_scalar: HbmTensor<bf16, Chip, m![1 # 8]>,
    pub mlp: MlpWeights,
}

pub(crate) enum Layer {
    Sliding(SlidingLayer),
    Full(FullLayer),
}

pub(crate) struct VisionWeights {
    pub patch_ln1_weight: HbmTensor<bf16, Chip, m![Rv]>,
    pub patch_ln1_bias: HbmTensor<bf16, Chip, m![Rv]>,
    pub patch_dense_weight: HbmTensor<bf16, Chip, m![Mv, Rv]>,
    pub patch_dense_bias: HbmTensor<bf16, Chip, m![Mv]>,
    pub patch_ln2_weight: HbmTensor<bf16, Chip, m![Mv]>,
    pub patch_ln2_bias: HbmTensor<bf16, Chip, m![Mv]>,
    pub pos_norm_weight: HbmTensor<bf16, Chip, m![Mv]>,
    pub pos_norm_bias: HbmTensor<bf16, Chip, m![Mv]>,
    pub embedding_projection_weight: HbmTensor<bf16, Chip, m![H, Ov]>,
    pub pos_embedding: Vec<bf16>,
}

pub(crate) struct AudioWeights {
    pub embedding_projection_weight: HbmTensor<bf16, Chip, m![H, Aa]>,
}

pub struct Model {
    pub(crate) model_dir: PathBuf,
    pub(crate) embedding_table: HbmTensor<bf16, Chip, m![W, H]>,
    pub(crate) final_norm: HbmTensor<bf16, Chip, m![H]>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) vision: VisionWeights,
    pub(crate) audio: AudioWeights,
}

async fn load_bf16<E: M>(
    ctx: &mut Context,
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
) -> Result<HbmTensor<bf16, Chip, E>, Box<dyn std::error::Error>> {
    let view = tensors.tensor(name)?;
    let host: HostTensor<bf16, E> = HostTensor::from_safetensors(&view).map_err(|error| format!("{name}: {error}"))?;
    Ok(host.to_hbm(&mut ctx.pdma).await)
}

async fn load_scalar_bf16(
    ctx: &mut Context,
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
) -> Result<HbmTensor<bf16, Chip, m![1 # 8]>, Box<dyn std::error::Error>> {
    let view = tensors.tensor(name)?;
    let host: HostTensor<bf16, m![1]> =
        HostTensor::from_safetensors(&view).map_err(|error| format!("{name}: {error}"))?;
    let value = host.into_vec()[0];
    Ok(HostTensor::<bf16, m![1 # 8]>::from_vec(vec![value; 8])
        .to_hbm(&mut ctx.pdma)
        .await)
}

async fn load_bf16_column<E: M>(
    ctx: &mut Context,
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
) -> Result<HbmTensor<bf16, Chip, E>, Box<dyn std::error::Error>> {
    let view = tensors.tensor(name)?;
    if view.dtype() != safetensors::Dtype::BF16 || view.shape() != [E::SIZE, 1] {
        return Err(format!(
            "{name} is {:?} {:?}, expected BF16 [{}, 1]",
            view.dtype(),
            view.shape(),
            E::SIZE
        )
        .into());
    }
    let host: HostTensor<bf16, E> = HostTensor::from_buf(view.data().to_vec());
    Ok(host.to_hbm(&mut ctx.pdma).await)
}

async fn load_f8<E: M>(
    ctx: &mut Context,
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
) -> Result<HbmTensor<f8e4m3, Chip, E>, Box<dyn std::error::Error>> {
    let view = tensors.tensor(name)?;
    let host: HostTensor<f8e4m3, E> =
        HostTensor::from_safetensors(&view).map_err(|error| format!("{name}: {error}"))?;
    Ok(host.to_hbm(&mut ctx.pdma).await)
}

async fn load_f4<E: M>(
    ctx: &mut Context,
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
    shape: [usize; 2],
) -> Result<HbmTensor<f4e2m1, Chip, E>, Box<dyn std::error::Error>> {
    let view = tensors.tensor(name)?;
    let packed = [shape[0], shape[1] / 2];
    if view.dtype() != safetensors::Dtype::U8 || view.shape() != packed {
        return Err(format!(
            "{name} is {:?} {:?}, expected U8 {packed:?}",
            view.dtype(),
            view.shape()
        )
        .into());
    }
    let host: HostTensor<f4e2m1, E> = HostTensor::from_buf(view.data().to_vec());
    Ok(host.to_hbm(&mut ctx.pdma).await)
}

fn read_f32(tensors: &safetensors::SafeTensors<'_>, name: &str) -> Result<f32, Box<dyn std::error::Error>> {
    let view = tensors.tensor(name)?;
    if view.dtype() != safetensors::Dtype::F32 || view.data().len() != 4 {
        return Err(format!("{name} is {:?}, not a scalar f32 tensor", view.dtype()).into());
    }
    Ok(f32::from_le_bytes(view.data().try_into()?))
}

fn reciprocal_global_scale(
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
) -> Result<f32, Box<dyn std::error::Error>> {
    let value = read_f32(tensors, name)?;
    if value == 0.0 || !value.is_finite() {
        return Err(format!("invalid NVFP4 global scale {name}: {value}").into());
    }
    Ok(1.0 / value)
}

async fn load_global_scale(
    ctx: &mut Context,
    tensors: &safetensors::SafeTensors<'_>,
    name: &str,
) -> Result<HbmTensor<f32, Chip, m![1]>, Box<dyn std::error::Error>> {
    let value = reciprocal_global_scale(tensors, name)?;
    Ok(HostTensor::<f32, m![1]>::from_vec(vec![value])
        .to_hbm(&mut ctx.pdma)
        .await)
}

pub fn model_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let root = std::env::var_os("RNGD_MODEL_DIR")
        .ok_or("RNGD_MODEL_DIR is not set; point it at the model directory or its parent")?;
    let root = PathBuf::from(root);
    let candidates = [
        root.clone(),
        root.join("Gemma4-12B-it"),
        root.join("gemma-4-12b-it-NVFP4"),
    ];
    candidates
        .into_iter()
        .find(|p| p.join("model.safetensors").is_file())
        .ok_or_else(|| format!("no model.safetensors found under {}", root.display()).into())
}

async fn load_mlp_projection<E: M, Es: M>(
    ctx: &mut Context,
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
    name: &str,
    shape: [usize; 2],
) -> Result<
    (
        HbmTensor<f4e2m1, Chip, E>,
        HbmTensor<f8e4m3, Chip, Es>,
        HbmTensor<f32, Chip, m![1]>,
    ),
    Box<dyn std::error::Error>,
> {
    Ok((
        load_f4(ctx, tensors, &format!("{prefix}.{name}.weight_packed"), shape).await?,
        load_f8(ctx, tensors, &format!("{prefix}.{name}.weight_scale")).await?,
        load_global_scale(ctx, tensors, &format!("{prefix}.{name}.weight_global_scale")).await?,
    ))
}

async fn load_mlp(
    ctx: &mut Context,
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
) -> Result<MlpWeights, Box<dyn std::error::Error>> {
    let (up_weight_packed, up_weight_scale, up_global_scale) =
        load_mlp_projection::<m![L, H], m![L, H / 16]>(ctx, tensors, prefix, "up_proj", [L::SIZE, H::SIZE]).await?;
    let (gate_weight_packed, gate_weight_scale, gate_global_scale) =
        load_mlp_projection::<m![L, H], m![L, H / 16]>(ctx, tensors, prefix, "gate_proj", [L::SIZE, H::SIZE]).await?;
    let (down_weight_packed, down_weight_scale, down_global_scale) =
        load_mlp_projection::<m![H, L], m![H, L / 16]>(ctx, tensors, prefix, "down_proj", [H::SIZE, L::SIZE]).await?;
    Ok(MlpWeights {
        up_weight_packed,
        gate_weight_packed,
        down_weight_packed,
        up_weight_scale,
        gate_weight_scale,
        down_weight_scale,
        up_global_scale,
        gate_global_scale,
        down_global_scale,
    })
}

async fn load_common(
    ctx: &mut Context,
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
) -> Result<
    (
        HbmTensor<bf16, Chip, m![H]>,
        HbmTensor<bf16, Chip, m![H]>,
        HbmTensor<bf16, Chip, m![H]>,
        MlpWeights,
    ),
    Box<dyn std::error::Error>,
> {
    Ok((
        load_bf16(ctx, tensors, &format!("{prefix}.pre_feedforward_layernorm.weight")).await?,
        load_bf16(ctx, tensors, &format!("{prefix}.post_feedforward_layernorm.weight")).await?,
        load_bf16(ctx, tensors, &format!("{prefix}.input_layernorm.weight")).await?,
        load_mlp(ctx, tensors, &format!("{prefix}.mlp")).await?,
    ))
}

async fn load_sliding(
    ctx: &mut Context,
    tensors: &safetensors::SafeTensors<'_>,
    layer: usize,
) -> Result<SlidingLayer, Box<dyn std::error::Error>> {
    let prefix = format!("model.language_model.layers.{layer}");
    let (pre_feedforward_norm, post_feedforward_norm, input_norm, mlp) = load_common(ctx, tensors, &prefix).await?;
    Ok(SlidingLayer {
        input_norm,
        q_weight: load_f8(ctx, tensors, &format!("{prefix}.self_attn.q_proj.weight")).await?,
        k_weight: load_f8(ctx, tensors, &format!("{prefix}.self_attn.k_proj.weight")).await?,
        v_weight: load_f8(ctx, tensors, &format!("{prefix}.self_attn.v_proj.weight")).await?,
        q_weight_scale: load_bf16_column(ctx, tensors, &format!("{prefix}.self_attn.q_proj.weight_scale")).await?,
        k_weight_scale: load_bf16_column(ctx, tensors, &format!("{prefix}.self_attn.k_proj.weight_scale")).await?,
        v_weight_scale: load_bf16_column(ctx, tensors, &format!("{prefix}.self_attn.v_proj.weight_scale")).await?,
        q_norm: load_bf16(ctx, tensors, &format!("{prefix}.self_attn.q_norm.weight")).await?,
        k_norm: load_bf16(ctx, tensors, &format!("{prefix}.self_attn.k_norm.weight")).await?,
        post_attention_norm: load_bf16(ctx, tensors, &format!("{prefix}.post_attention_layernorm.weight")).await?,
        o_weight: load_f8(ctx, tensors, &format!("{prefix}.self_attn.o_proj.weight")).await?,
        o_weight_scale: load_bf16_column(ctx, tensors, &format!("{prefix}.self_attn.o_proj.weight_scale")).await?,
        pre_feedforward_norm,
        post_feedforward_norm,
        layer_scalar: load_scalar_bf16(ctx, tensors, &format!("{prefix}.layer_scalar")).await?,
        mlp,
    })
}

async fn load_full(
    ctx: &mut Context,
    tensors: &safetensors::SafeTensors<'_>,
    layer: usize,
) -> Result<FullLayer, Box<dyn std::error::Error>> {
    let prefix = format!("model.language_model.layers.{layer}");
    let (pre_feedforward_norm, post_feedforward_norm, input_norm, mlp) = load_common(ctx, tensors, &prefix).await?;
    Ok(FullLayer {
        input_norm,
        q_weight: load_f8(ctx, tensors, &format!("{prefix}.self_attn.q_proj.weight")).await?,
        k_weight: load_f8(ctx, tensors, &format!("{prefix}.self_attn.k_proj.weight")).await?,
        q_weight_scale: load_bf16_column(ctx, tensors, &format!("{prefix}.self_attn.q_proj.weight_scale")).await?,
        k_weight_scale: load_bf16_column(ctx, tensors, &format!("{prefix}.self_attn.k_proj.weight_scale")).await?,
        q_norm: load_bf16(ctx, tensors, &format!("{prefix}.self_attn.q_norm.weight")).await?,
        k_norm: load_bf16(ctx, tensors, &format!("{prefix}.self_attn.k_norm.weight")).await?,
        post_attention_norm: load_bf16(ctx, tensors, &format!("{prefix}.post_attention_layernorm.weight")).await?,
        o_weight: load_f8(ctx, tensors, &format!("{prefix}.self_attn.o_proj.weight")).await?,
        o_weight_scale: load_bf16_column(ctx, tensors, &format!("{prefix}.self_attn.o_proj.weight_scale")).await?,
        pre_feedforward_norm,
        post_feedforward_norm,
        layer_scalar: load_scalar_bf16(ctx, tensors, &format!("{prefix}.layer_scalar")).await?,
        mlp,
    })
}

async fn load_vision(
    ctx: &mut Context,
    tensors: &safetensors::SafeTensors<'_>,
) -> Result<VisionWeights, Box<dyn std::error::Error>> {
    let pos_embedding_view = tensors.tensor("model.vision_embedder.pos_embedding")?;
    let pos_embedding_host: HostTensor<bf16, m![Pv, Dummy2, Mv]> =
        HostTensor::from_safetensors(&pos_embedding_view).map_err(|error| format!("pos_embedding: {error}"))?;

    Ok(VisionWeights {
        patch_ln1_weight: load_bf16(ctx, tensors, "model.vision_embedder.patch_ln1.weight").await?,
        patch_ln1_bias: load_bf16(ctx, tensors, "model.vision_embedder.patch_ln1.bias").await?,
        patch_dense_weight: load_bf16(ctx, tensors, "model.vision_embedder.patch_dense.weight").await?,
        patch_dense_bias: load_bf16(ctx, tensors, "model.vision_embedder.patch_dense.bias").await?,
        patch_ln2_weight: load_bf16(ctx, tensors, "model.vision_embedder.patch_ln2.weight").await?,
        patch_ln2_bias: load_bf16(ctx, tensors, "model.vision_embedder.patch_ln2.bias").await?,
        pos_norm_weight: load_bf16(ctx, tensors, "model.vision_embedder.pos_norm.weight").await?,
        pos_norm_bias: load_bf16(ctx, tensors, "model.vision_embedder.pos_norm.bias").await?,
        embedding_projection_weight: load_bf16(ctx, tensors, "model.embed_vision.embedding_projection.weight").await?,
        pos_embedding: pos_embedding_host.into_vec(),
    })
}

async fn load_audio(
    ctx: &mut Context,
    tensors: &safetensors::SafeTensors<'_>,
) -> Result<AudioWeights, Box<dyn std::error::Error>> {
    Ok(AudioWeights {
        embedding_projection_weight: load_bf16(ctx, tensors, "model.embed_audio.embedding_projection.weight").await?,
    })
}

pub async fn load_model(ctx: &mut Context) -> Result<Model, Box<dyn std::error::Error>> {
    let model_dir = model_dir()?;
    let model_path = model_dir.join("model.safetensors");
    let file = File::open(&model_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = safetensors::SafeTensors::deserialize(&mmap)?;

    let embedding_table = load_bf16(ctx, &tensors, "model.language_model.embed_tokens.weight").await?;
    let final_norm = load_bf16(ctx, &tensors, "model.language_model.norm.weight").await?;
    let vision = load_vision(ctx, &tensors).await?;
    let audio = load_audio(ctx, &tensors).await?;

    let mut layers = Vec::with_capacity(LAYERS);
    for layer in 0..LAYERS {
        if layer % 6 == 5 {
            layers.push(Layer::Full(load_full(ctx, &tensors, layer).await?));
        } else {
            layers.push(Layer::Sliding(load_sliding(ctx, &tensors, layer).await?));
        }
    }

    Ok(Model {
        model_dir,
        embedding_table,
        final_norm,
        layers,
        vision,
        audio,
    })
}

pub fn tokenizer_path(model: &Model) -> PathBuf {
    tokenizer_path_in(&model.model_dir)
}

pub fn tokenizer_path_in(model_dir: &Path) -> PathBuf {
    model_dir.join("tokenizer.json")
}

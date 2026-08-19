
use furiosa_opt_std::prelude::*;

use crate::axes::*;
use crate::device::layout::{self, Cluster, Replicated, Slice};
use crate::device::{full, shared, sliding};
use crate::{Chip, EMBED_SCALE, LOGIT_SOFTCAP};

#[device(chip = 1)]
pub fn embed_token(
    ctx: &mut Context,
    embedding_table: &HbmTensor<bf16, Chip, m![W, H]>,
    offset: &HbmTensor<i32, Chip, m![1]>,
    out: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let row: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = embedding_table.dma_gather_scaled(offset);
    let row: DmTensor<bf16, Chip, Cluster, m![H / 240, 1 # 16], m![H % 240]> = row.to_dm(&mut ctx.tdma);

    let result: DmTensor<bf16, Chip, Cluster, m![H / 240, 1 # 16], m![H % 240]> = ctx
        .main
        .begin(row.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), EMBED_SCALE)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    result.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
}

#[device(chip = 1)]
pub fn copy_hidden_state(ctx: &mut Context, x: &HbmTensor<bf16, Chip, m![H]>, out: &mut HbmTensor<bf16, Chip, m![H]>) {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = x.to_dm(&mut ctx.tdma);
    x.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
}

#[device(chip = 1)]
pub fn sliding_project_qkv(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Ds]>,
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = x.to_dm(&mut ctx.tdma);
    let x = shared::rmsnorm::normalize(ctx, &x, input_rms_weight);

    let x: DmTensor<bf16, Chip, Cluster, Replicated, m![H]> = layout::broadcast_hidden(ctx, &x);

    let q: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> =
        sliding::projection::project_query(ctx, &x, q_weight, q_weight_scale);
    let (k, v) = sliding::projection::project_key_value(ctx, &x, k_weight, v_weight, k_weight_scale, v_weight_scale);

    let q: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> =
        sliding::rmsnorm::normalize_query(ctx, &q, q_rms_weight);
    let k: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> = sliding::rmsnorm::normalize_key(ctx, &k, k_rms_weight);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> = sliding::rmsnorm::normalize_value(ctx, &v);

    let (q, k) = sliding::rope::apply_rope(ctx, &q, &k, rope_offset, cos, sin);

    q.view().to_hbm_view(&mut ctx.tdma, q_out.view_mut());
    k.dma_scatter::<m![1], _, _>(kv_offset, k_cache);
    v.dma_scatter::<m![1], _, _>(kv_offset, v_cache);
}

#[device(chip = 1)]
pub fn full_project_qkv(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    q_weight: &HbmTensor<f8e4m3, Chip, m![Qf, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Pf, H]>,
    q_weight_scale: &HbmTensor<bf16, Chip, m![Qf]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Pf]>,
    input_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    q_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    k_rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Df]>,
    sin: &HbmTensor<bf16, Chip, m![E, Df]>,
    k_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![Tf, Df]>,
    q_out: &mut HbmTensor<bf16, Chip, m![Gf, Df]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = x.to_dm(&mut ctx.tdma);
    let x = shared::rmsnorm::normalize(ctx, &x, input_rms_weight);

    let x: DmTensor<bf16, Chip, Cluster, Replicated, m![H]> = layout::broadcast_hidden(ctx, &x);

    let q: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> =
        full::projection::project_query(ctx, &x, q_weight, q_weight_scale);
    let k_raw: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> =
        full::projection::project_key(ctx, &x, k_weight, k_weight_scale);

    let q: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> = full::rmsnorm::normalize_query(ctx, &q, q_rms_weight);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = full::rmsnorm::normalize_value(ctx, &k_raw);
    let k: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = full::rmsnorm::normalize_key(ctx, &k_raw, k_rms_weight);

    let (q, k) = full::rope::apply_rope(ctx, &q, &k, rope_offset, cos, sin);

    q.view().to_hbm_view(&mut ctx.tdma, q_out.view_mut());
    k.dma_scatter::<m![1], _, _>(kv_offset, k_cache);
    v.dma_scatter::<m![1], _, _>(kv_offset, v_cache);
}

#[device(chip = 1)]
pub fn sliding_attention(
    ctx: &mut Context,
    q: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    mask: &HbmTensor<f32, Chip, m![Ts]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
) {
    sliding::attention::attend(ctx, q, k, v, mask, out_hbm);
}

#[device(chip = 1)]
pub fn sliding_attention_output(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    o_weight_scale: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Qs]> = unsafe { x.reshape() };
    let x: DmTensor<bf16, Chip, Cluster, Replicated, m![Qs]> = layout::broadcast_sliding_heads(ctx, &x);

    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> =
        sliding::projection::project_output(ctx, &x, o_weight, o_weight_scale);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = shared::rmsnorm::normalize(ctx, &x, post_attn_rms_weight);

    let residual: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = residual_hbm.to_dm(&mut ctx.tdma);
    let residual: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = shared::residual::add(ctx, &x, &residual);
    residual.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());
}

#[device(chip = 1)]
pub fn full_attention_first_page(
    ctx: &mut Context,
    q: &HbmTensor<bf16, Chip, m![Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    mask: &HbmTensor<f32, Chip, m![Tf]>,
    running_max: &mut HbmTensor<f32, Chip, m![Gf]>,
    running_sum: &mut HbmTensor<f32, Chip, m![Gf]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![Gf, Df]>,
) {
    full::attention::attend_first_page(ctx, q, k, v, mask, running_max, running_sum, out_hbm);
}

#[device(chip = 1)]
pub fn full_attention_page(
    ctx: &mut Context,
    q: &HbmTensor<bf16, Chip, m![Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    mask: &HbmTensor<f32, Chip, m![Tf]>,
    running_max: &mut HbmTensor<f32, Chip, m![Gf]>,
    running_sum: &mut HbmTensor<f32, Chip, m![Gf]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![Gf, Df]>,
) {
    full::attention::attend_next_page(ctx, q, k, v, mask, running_max, running_sum, out_hbm);
}

#[device(chip = 1)]
pub fn full_attention_output(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![Gf, Df]>,
    running_sum: &HbmTensor<f32, Chip, m![Gf]>,
    post_attn_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    o_weight: &HbmTensor<f8e4m3, Chip, m![H, Qf]>,
    o_weight_scale: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> =
        full::attention::divide_by_softmax_sum(ctx, &x, running_sum);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Qf]> = unsafe { x.reshape() };
    let x: DmTensor<bf16, Chip, Cluster, Replicated, m![Qf]> = layout::broadcast_full_heads(ctx, &x);

    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> =
        full::projection::project_output(ctx, &x, o_weight, o_weight_scale);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = shared::rmsnorm::normalize(ctx, &x, post_attn_rms_weight);

    let residual: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = residual_hbm.to_dm(&mut ctx.tdma);
    let residual: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = shared::residual::add(ctx, &x, &residual);
    residual.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());
}

#[device(chip = 1)]
pub fn decoder_feedforward(
    ctx: &mut Context,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
    pre_ff_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
    post_ff_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) {
    let residual: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = residual_hbm.to_dm(&mut ctx.tdma);
    let x = shared::rmsnorm::normalize(ctx, &residual, pre_ff_rms_weight);

    let x: DmTensor<bf16, Chip, Cluster, Replicated, m![H]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = shared::mlp::feedforward(
        ctx,
        x,
        up_weight_packed,
        gate_weight_packed,
        down_weight_packed,
        up_weight_scale,
        gate_weight_scale,
        down_weight_scale,
        up_global_scale,
        gate_global_scale,
        down_global_scale,
    );

    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = shared::rmsnorm::normalize(ctx, &x, post_ff_rms_weight);
    let residual: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = shared::residual::add(ctx, &x, &residual);
    let residual: DmTensor<bf16, Chip, Cluster, Slice, m![H]> =
        shared::residual::scale_by_layer_gate(ctx, &residual, layer_scalar);
    residual.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());
}

#[device(chip = 1)]
pub fn final_norm_and_logits(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    lm_head_weight: &HbmTensor<bf16, Chip, m![W, H]>,
    out: &mut HbmTensor<bf16, Chip, m![W]>,
) {
    let x: DmTensor<bf16, Chip, shared::lm_head::Cluster, Slice, m![H]> = input.to_dm(&mut ctx.tdma);
    let x = shared::rmsnorm::normalize(ctx, &x, rms_weight);

    let logits = shared::lm_head::logits(ctx, &x, lm_head_weight);

    let scaled: DmTensor<
        f32,
        Chip,
        shared::lm_head::Cluster,
        shared::lm_head::LogitSlices,
        shared::lm_head::LogitsPerSlice,
    > = ctx
        .main
        .begin(logits.view())
        .fetch::<m![W / 16 % 32], m![W % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![W / 8 % 64], m![W % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![W / 4 % 128], m![W % 4]>()
        .vector_fp_div(LOGIT_SOFTCAP)
        .vector_widen_concat::<m![W / 8 % 64], m![W % 8]>()
        .vector_final()
        .commit_trim::<m![W % 8]>()
        .commit();

    let capped: DmTensor<
        bf16,
        Chip,
        shared::lm_head::Cluster,
        shared::lm_head::LogitSlices,
        shared::lm_head::LogitsPerSlice,
    > = ctx
        .main
        .begin(scaled.view())
        .fetch::<m![W / 8 % 64], m![W % 8]>()
        .collect::<m![W / 8 % 64], m![W % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![W / 4 % 128], m![W % 4]>()
        .vector_fp_unary(FpUnaryOp::Tanh)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), LOGIT_SOFTCAP)
        .vector_widen_concat::<m![W / 8 % 64], m![W % 8]>()
        .vector_final()
        .cast::<bf16, m![W % 8 # 16]>()
        .commit_trim::<m![W % 8]>()
        .commit();

    capped.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
}

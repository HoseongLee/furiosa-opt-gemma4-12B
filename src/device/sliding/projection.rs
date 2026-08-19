
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Ds, Gs, H, Ns, Ps, Qs};
use crate::device::layout::{Cluster, Replicated, Slice};

pub(crate) fn project_query(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Replicated, m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> {
    type QueryRows = m![Qs / 16];

    let x: DmTensorView<'_, bf16, Chip, Cluster, QueryRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, Cluster, QueryRows, m![1], m![H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![1], m![H]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let weight_f8: DmTensor<f8e4m3, Chip, Cluster, QueryRows, m![Qs % 16, H]> = weight.to_dm(&mut ctx.tdma);
    let weight_dm: DmTensor<bf16, Chip, Cluster, QueryRows, m![Qs % 16, H]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Qs % 16, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Qs % 16, H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();

    let contraction: DmTensor<bf16, Chip, Cluster, QueryRows, m![Qs % 16]> = ctx
        .main
        .begin(weight_dm.view())
        .fetch::<m![Qs % 16, H / 16], m![H % 16]>()
        .collect::<m![Qs % 16, H / 16], m![H % 16]>()
        .contract_outer::<m![Qs % 16, H / 32], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 16]>()
        .contract_lane::<m![Qs % 16], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 4 % 4], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, Cluster, QueryRows, m![Qs % 16]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, QueryRows, m![Qs % 16]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 2], m![Qs % 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, Cluster, QueryRows, m![Qs % 16]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Qs % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 2], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 4], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![Qs / 8 % 2], m![Qs % 8]>()
        .vector_final()
        .cast::<bf16, m![Qs % 8 # 16]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();

    let output: DmTensor<bf16, Chip, Cluster, Slice, m![Qs]> = ctx
        .main
        .begin(scaled.view())
        .fetch::<m![1], m![Qs % 16]>()
        .switch::<Slice, m![Qs / 16]>(SwitchConfig::Broadcast1 { slice1: 256, slice0: 1 })
        .collect::<m![Qs / 16], m![Qs % 16]>()
        .commit_trim::<m![Qs % 16]>()
        .commit();

    unsafe { output.reshape() }
}

type KvRows = m![Ps / 8];

fn project_one_kv_matrix(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, Cluster, KvRows, m![1], m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> {
    let weight_f8: DmTensor<f8e4m3, Chip, Cluster, KvRows, m![Ps % 8, H]> = weight.to_dm(&mut ctx.tdma);
    let weight_dm: DmTensor<bf16, Chip, Cluster, KvRows, m![Ps % 8, H]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ps % 8, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Ps % 8, H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();

    let contraction: DmTensor<bf16, Chip, Cluster, KvRows, m![Ps % 8]> = ctx
        .main
        .begin(weight_dm.view())
        .fetch::<m![Ps % 8, H / 16], m![H % 16]>()
        .collect::<m![Ps % 8, H / 16], m![H % 16]>()
        .contract_outer::<m![Ps % 8, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 8]>()
        .contract_lane::<m![Ps % 8], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Ps / 4 % 2], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, Cluster, KvRows, m![Ps % 8]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, KvRows, m![Ps % 8]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Ps % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, Cluster, KvRows, m![Ps % 8]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Ps % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ps / 4 % 2], m![Ps % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![1], m![Ps % 8]>()
        .vector_final()
        .cast::<bf16, m![Ps % 8 # 16]>()
        .commit_trim::<m![Ps % 8]>()
        .commit();

    ctx.main
        .begin(scaled.view())
        .fetch::<m![1], m![Ps % 8 # 16]>()
        .switch::<Slice, m![Ps / 8]>(SwitchConfig::Broadcast1 { slice1: 256, slice0: 1 })
        .collect::<m![Ps / 8], m![Ps % 8 # 16]>()
        .commit_trim::<m![Ps % 8]>()
        .commit()
}

pub(crate) fn project_key_value(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Replicated, m![H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
) {
    let x: DmTensorView<'_, bf16, Chip, Cluster, KvRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, Cluster, KvRows, m![1], m![H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let k: DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> = project_one_kv_matrix(ctx, &x_trf, k_weight, k_weight_scale);
    let v: DmTensor<bf16, Chip, Cluster, Slice, m![Ps]> = project_one_kv_matrix(ctx, &x_trf, v_weight, v_weight_scale);

    (unsafe { k.reshape() }, unsafe { v.reshape() })
}

pub(crate) fn project_output(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Replicated, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    const CHUNK: usize = 1024;

    let p0 = output_partial(ctx, x, weight, 0);
    let p1 = output_partial(ctx, x, weight, CHUNK);
    let p2 = output_partial(ctx, x, weight, 2 * CHUNK);
    let p3 = output_partial(ctx, x, weight, 3 * CHUNK);

    let p01 = add_partials(ctx, &p0, &p1);
    let p23 = add_partials(ctx, &p2, &p3);
    let result = add_partials(ctx, &p01, &p23);
    let result = apply_output_channel_scale(ctx, &result, weight_scale);
    result.to_dm(&mut ctx.tdma)
}

type HiddenRows = m![H / 120, 1 # 8];

fn output_partial(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Replicated, m![Qs]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    offset: usize,
) -> DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]> {
    let x: DmTensorView<'_, bf16, Chip, Cluster, HiddenRows, m![Qs]> = unsafe { x.view().reshape() };
    let x = x.tile::<m![Qs], 1024, m![Qs = 1024 # 4096]>(offset);
    let x_trf: TrfTensor<bf16, Chip, Cluster, HiddenRows, m![1], m![Qs = 1024]> = ctx
        .sub
        .begin(x)
        .fetch::<m![1], m![Qs = 1024]>()
        .collect::<m![Qs = 1024 / 16], m![Qs = 1024 % 16]>()
        .to_trf();
    let weight_f8: DmTensor<f8e4m3, Chip, Cluster, HiddenRows, m![H % 120, Qs = 1024]> = weight
        .view()
        .tile::<m![Qs], 1024, m![H, Qs = 1024 # 4096]>(offset)
        .to_dm(&mut ctx.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120, Qs = 1024]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![H % 120, Qs = 1024 / 32], m![Qs = 1024 % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 120, Qs = 1024 / 16], m![Qs = 1024 % 16]>()
        .commit_trim::<m![Qs = 1024 % 16]>()
        .commit();
    ctx.main
        .begin(weight.view())
        .fetch::<m![H % 120, Qs = 1024 / 16], m![Qs = 1024 % 16]>()
        .collect::<m![H % 120, Qs = 1024 / 16], m![Qs = 1024 % 16]>()
        .contract_outer::<m![H % 120, Qs = 1024 / 32], m![Qs = 1024 % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120]>()
        .contract_lane::<m![H % 120], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H / 4 % 30], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit()
}

fn add_partials(
    ctx: &mut Context,
    left: &DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]>,
    right: &DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]>,
) -> DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]> {
    let left: VrfTensor<f32, Chip, Cluster, HiddenRows, m![H % 120]> = ctx
        .sub
        .begin(left.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();
    ctx.main
        .begin(right.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &left)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

fn apply_output_channel_scale(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]> {
    let weight_scale: DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, HiddenRows, m![H % 120]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();

    ctx.main
        .begin(x.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

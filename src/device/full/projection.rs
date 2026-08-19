
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Df, Gf, H, Pf, Qf};
use crate::device::layout::{Cluster, Replicated, Slice};

pub(crate) fn project_query(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Replicated, m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qf, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qf]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> {
    type QueryRows = m![Qf / 32];

    let x: DmTensorView<'_, bf16, Chip, Cluster, QueryRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, Cluster, QueryRows, m![1], m![H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![1], m![H]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let weight_f8: DmTensor<f8e4m3, Chip, Cluster, QueryRows, m![Qf % 32, H]> = weight.to_dm(&mut ctx.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, QueryRows, m![Qf % 32, H]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Qf % 32, H / 32], m![H % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Qf % 32, H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();

    let result: DmTensor<bf16, Chip, Cluster, QueryRows, m![Qf % 32]> = ctx
        .main
        .begin(weight.view())
        .fetch::<m![Qf % 32, H / 16], m![H % 16]>()
        .collect::<m![Qf % 32, H / 16], m![H % 16]>()
        .contract_outer::<m![Qf % 32, H / 32], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qf % 32]>()
        .contract_lane::<m![Qf % 32], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qf / 4 % 8], m![Qf % 4 # 16]>()
        .commit_trim::<m![Qf % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, Cluster, QueryRows, m![Qf % 32]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, QueryRows, m![Qf % 32]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Qf % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![Qf / 8 % 4], m![Qf % 8]>()
        .to_vrf();

    let result: DmTensor<bf16, Chip, Cluster, QueryRows, m![Qf % 32]> = ctx
        .main
        .begin(result.view())
        .fetch::<m![1], m![Qf % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![Qf / 8 % 4], m![Qf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qf / 4 % 8], m![Qf % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![Qf / 8 % 4], m![Qf % 8]>()
        .vector_final()
        .cast::<bf16, m![Qf % 8 # 16]>()
        .commit_trim::<m![Qf % 8]>()
        .commit();

    let result: DmTensor<bf16, Chip, Cluster, Slice, m![Qf]> = ctx
        .main
        .begin(result.view())
        .fetch::<m![1], m![Qf % 32]>()
        .switch::<Slice, m![Qf / 32]>(SwitchConfig::Broadcast1 { slice1: 256, slice0: 1 })
        .collect::<m![Qf / 16], m![Qf % 16]>()
        .commit_trim::<m![Qf % 16]>()
        .commit();

    unsafe { result.reshape() }
}

type KvRows = m![Pf / 8, 1 # 4];

pub(crate) fn project_key(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Replicated, m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Pf, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Pf]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Df]> {
    let x: DmTensorView<'_, bf16, Chip, Cluster, KvRows, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, Cluster, KvRows, m![1], m![H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let weight_f8: DmTensor<f8e4m3, Chip, Cluster, KvRows, m![Pf % 8, H]> = weight.to_dm(&mut ctx.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, KvRows, m![Pf % 8, H]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Pf % 8, H / 16], m![H % 16]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Pf % 8, H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    let result: DmTensor<bf16, Chip, Cluster, KvRows, m![Pf % 8]> = ctx
        .main
        .begin(weight.view())
        .fetch::<m![Pf % 8, H / 16], m![H % 16]>()
        .collect::<m![Pf % 8, H / 16], m![H % 16]>()
        .contract_outer::<m![Pf % 8, H / 32], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Pf % 8]>()
        .contract_lane::<m![Pf % 8], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Pf / 4 % 2], m![Pf % 4 # 16]>()
        .commit_trim::<m![Pf % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, Cluster, KvRows, m![Pf % 8]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, KvRows, m![Pf % 8]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Pf % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Pf % 8]>()
        .to_vrf();

    let result: DmTensor<bf16, Chip, Cluster, KvRows, m![Pf % 8]> = ctx
        .main
        .begin(result.view())
        .fetch::<m![1], m![Pf % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Pf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Pf / 4 % 2], m![Pf % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![1], m![Pf % 8]>()
        .vector_final()
        .cast::<bf16, m![Pf % 8 # 16]>()
        .commit_trim::<m![Pf % 8]>()
        .commit();

    let result: DmTensor<bf16, Chip, Cluster, Slice, m![Pf]> = ctx
        .main
        .begin(result.view())
        .fetch::<m![1], m![Pf % 8 # 16]>()
        .switch::<Slice, m![Pf / 8]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 4 })
        .collect::<m![Pf / 8], m![Pf % 8 # 16]>()
        .commit_trim::<m![Pf % 8]>()
        .commit();

    unsafe { result.reshape() }
}

pub(crate) fn project_output(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Replicated, m![Qf]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qf]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    const CHUNK: usize = 1024;

    let p0 = output_partial(ctx, x, weight, 0);
    let p1 = output_partial(ctx, x, weight, CHUNK);
    let p2 = output_partial(ctx, x, weight, 2 * CHUNK);
    let p3 = output_partial(ctx, x, weight, 3 * CHUNK);
    let p4 = output_partial(ctx, x, weight, 4 * CHUNK);
    let p5 = output_partial(ctx, x, weight, 5 * CHUNK);
    let p6 = output_partial(ctx, x, weight, 6 * CHUNK);
    let p7 = output_partial(ctx, x, weight, 7 * CHUNK);

    let p01 = add_partials(ctx, &p0, &p1);
    let p23 = add_partials(ctx, &p2, &p3);
    let p45 = add_partials(ctx, &p4, &p5);
    let p67 = add_partials(ctx, &p6, &p7);
    let p0123 = add_partials(ctx, &p01, &p23);
    let p4567 = add_partials(ctx, &p45, &p67);
    let result = add_partials(ctx, &p0123, &p4567);
    let result = apply_output_channel_scale(ctx, &result, weight_scale);

    result.to_dm(&mut ctx.tdma)
}

type HiddenRows = m![H / 120, 1 # 8];

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

fn output_partial(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Replicated, m![Qf]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qf]>,
    offset: usize,
) -> DmTensor<bf16, Chip, Cluster, m![H / 120, 1 # 8], m![H % 120]> {
    type HiddenRows = m![H / 120, 1 # 8];

    let x: DmTensorView<'_, bf16, Chip, Cluster, HiddenRows, m![Qf]> = unsafe { x.view().reshape() };
    let x_half = x.tile::<m![Qf], 1024, m![Qf = 1024 # 8192]>(offset);
    let x_trf: TrfTensor<bf16, Chip, Cluster, HiddenRows, m![1], m![Qf = 1024]> = ctx
        .sub
        .begin(x_half)
        .fetch::<m![1], m![Qf = 1024]>()
        .collect::<m![Qf = 1024 / 16], m![Qf = 1024 % 16]>()
        .to_trf();

    let weight_f8: DmTensor<f8e4m3, Chip, Cluster, HiddenRows, m![H % 120, Qf = 1024]> = weight
        .view()
        .tile::<m![Qf], 1024, m![H, Qf = 1024 # 8192]>(offset)
        .to_dm(&mut ctx.tdma);
    let weight: DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120, Qf = 1024]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![H % 120, Qf = 1024 / 32], m![Qf = 1024 % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120, Qf = 1024 / 8], m![Qf = 1024 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120, Qf = 1024 / 4], m![Qf = 1024 % 4]>()
        .vector_widen_concat::<m![H % 120, Qf = 1024 / 8], m![Qf = 1024 % 8]>()
        .vector_final()
        .cast::<bf16, m![Qf = 1024 % 8 # 16]>()
        .commit_trim::<m![Qf = 1024 % 8]>()
        .commit();

    ctx.main
        .begin(weight.view())
        .fetch::<m![H % 120, Qf = 1024 / 16], m![Qf = 1024 % 16]>()
        .collect::<m![H % 120, Qf = 1024 / 16], m![Qf = 1024 % 16]>()
        .contract_outer::<m![H % 120, Qf = 1024 / 32], m![Qf = 1024 % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120]>()
        .contract_lane::<m![H % 120], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H / 4 % 30], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit()
}

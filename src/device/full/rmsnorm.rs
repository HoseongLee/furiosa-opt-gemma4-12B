
use furiosa_opt_std::prelude::*;

use crate::axes::{Df, Gf};
use crate::device::layout::{Cluster, Slice};
use crate::{Chip, EPS};

const DF_F32: f32 = Df::SIZE as f32;

pub(crate) fn normalize_query<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> {
    let mean_square: DmTensor<f32, Chip, Cluster, Slice, m![Gf]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Gf, Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gf, Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gf, Df / 4], m![Df % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Df, m![Gf], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DF_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .transpose::<m![Gf / 2], m![Gf % 2 # 8]>()
        .commit_trim::<m![Gf % 2]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, Slice, m![Gf]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![Gf / 8], m![Gf % 8]>()
        .collect::<m![Gf / 8], m![Gf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gf / 4], m![Gf % 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_concat::<m![Gf / 8], m![Gf % 8]>()
        .vector_final()
        .commit_trim::<m![Gf % 8]>()
        .commit();

    let weight_vrf = load_norm_weight::<Cluster, Slice>(ctx, rms_weight);

    let rms_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Gf]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![Gf / 8], m![Gf % 8]>()
        .collect::<m![Gf / 8], m![Gf % 8]>()
        .to_vrf();

    ctx.main
        .begin(x.view())
        .fetch::<m![Gf, Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gf, Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gf, Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![Gf, Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}

fn load_norm_weight<Cluster: M, Slice: M>(
    ctx: &mut Context,
    rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
) -> VrfTensor<f32, Chip, Cluster, Slice, m![Df]> {
    let weight_dm: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = rms_weight.to_dm(&mut ctx.tdma);

    ctx.sub
        .begin(weight_dm.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .to_vrf()
}

fn root_mean_square<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Df]>,
) -> VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]> {
    let mean_square: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Df / 4], m![Df % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Df, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DF_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .transpose::<m![1], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    ctx.sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

fn scale_by_rms_and_weight<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Df]>,
    rms_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]>,
    weight_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![Df]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Df]> {
    ctx.main
        .begin(x.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight_vrf)
        .vector_fp_div(rms_vrf)
        .vector_widen_concat::<m![Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}

fn scale_by_rms<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Df]>,
    rms_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Df]> {
    ctx.main
        .begin(x.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Df / 4], m![Df % 4]>()
        .vector_fp_div(rms_vrf)
        .vector_widen_concat::<m![Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}

pub(crate) fn normalize_key(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Df]>,
    rms_weight: &HbmTensor<bf16, Chip, m![Df]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Df]> {
    let rms_vrf = root_mean_square(ctx, x);
    let weight_vrf = load_norm_weight::<Cluster, Slice>(ctx, rms_weight);

    scale_by_rms_and_weight(ctx, x, &rms_vrf, &weight_vrf)
}

pub(crate) fn normalize_value(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Df]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Df]> {
    let rms_vrf = root_mean_square(ctx, x);

    scale_by_rms(ctx, x, &rms_vrf)
}

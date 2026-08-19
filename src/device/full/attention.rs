
use furiosa_opt_std::prelude::*;

use crate::axes::{Df, Gf, Tf};
use crate::device::layout::{Cluster, Slice};
use crate::{Chip, EPS};

type PerQueryHead = m![Gf, 1 # 16];

fn per_head_to_vrf<Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<f32, Chip, Cluster, Slice, m![Gf]>,
) -> VrfTensor<f32, Chip, Cluster, Slice, m![Gf]> {
    ctx.sub
        .begin(x.view())
        .fetch::<m![Gf / 8], m![Gf % 8]>()
        .collect::<m![Gf / 8], m![Gf % 8]>()
        .to_vrf()
}

fn query_key_scores(
    ctx: &mut Context,
    q: &HbmTensor<bf16, Chip, m![Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Tf, Df]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Tf]> {
    type KvRowsAcrossSlices = m![Tf / 8, 1 # 4];

    let q: DmTensor<bf16, Chip, Cluster, KvRowsAcrossSlices, m![Gf, Df]> = q.to_dm(&mut ctx.tdma);
    let k: DmTensor<bf16, Chip, Cluster, KvRowsAcrossSlices, m![Tf % 8, Df]> = k.to_dm(&mut ctx.tdma);
    let k_trf: TrfTensor<bf16, Chip, Cluster, KvRowsAcrossSlices, m![Tf % 8], m![Df]> = ctx
        .sub
        .begin(k.view())
        .fetch::<m![Tf % 8, Df / 16], m![Df % 16]>()
        .collect::<m![Tf % 8, Df / 16], m![Df % 16]>()
        .to_trf();

    let qk: DmTensor<bf16, Chip, Cluster, KvRowsAcrossSlices, m![Gf, Tf % 8]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![Gf, Df / 16], m![Df % 16]>()
        .collect::<m![Gf, Df / 16], m![Df % 16]>()
        .contract_outer::<m![Gf, Df / 32], m![Df % 32], _, _, _>(&k_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Gf]>()
        .contract_lane::<m![Gf], m![Tf % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Tf % 8 # 16]>()
        .commit_trim::<m![Tf % 8]>()
        .commit();

    ctx.main
        .begin(qk.view())
        .fetch::<m![Gf], m![Tf % 8 # 16]>()
        .switch::<Slice, m![Gf, Tf / 8]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 4 })
        .collect::<m![Gf, Tf / 8], m![Tf % 8 # 16]>()
        .commit_trim::<m![Tf % 8]>()
        .commit()
}

fn row_max(
    ctx: &mut Context,
    qk: &DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Tf]>,
) -> DmTensor<f32, Chip, Cluster, Slice, m![Gf]> {
    ctx.main
        .begin(qk.view())
        .fetch::<m![Gf, Tf / 16], m![Tf % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gf, Tf / 8], m![Tf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gf, Tf / 4], m![Tf % 4]>()
        .vector_intra_slice_reduce::<Tf, m![Gf], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![Gf / 2], m![Gf % 2 # 8]>()
        .commit_trim::<m![Gf % 2]>()
        .commit()
}

fn masked_exp(
    ctx: &mut Context,
    qk: &DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Tf]>,
    max_vrf: &VrfTensor<f32, Chip, Cluster, Slice, m![Gf]>,
    mask: &HbmTensor<f32, Chip, m![Tf]>,
) -> DmTensor<f32, Chip, Cluster, Slice, m![Gf, Tf]> {
    let exp: DmTensor<f32, Chip, Cluster, Slice, m![Gf, Tf]> = ctx
        .main
        .begin(qk.view())
        .fetch::<m![Gf, Tf / 16], m![Tf % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gf, Tf / 8], m![Tf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gf, Tf / 4], m![Tf % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, max_vrf)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![Gf, Tf / 8], m![Tf % 8]>()
        .vector_final()
        .commit_trim::<m![Tf % 8]>()
        .commit();

    let mask: DmTensor<f32, Chip, Cluster, Slice, m![Tf]> = mask.to_dm(&mut ctx.tdma);
    let mask_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Tf]> = ctx
        .sub
        .begin(mask.view())
        .fetch::<m![Tf / 8], m![Tf % 8]>()
        .collect::<m![Tf / 8], m![Tf % 8]>()
        .to_vrf();

    ctx.main
        .begin(exp.view())
        .fetch::<m![Gf, Tf / 8], m![Tf % 8]>()
        .collect::<m![Gf, Tf / 8], m![Tf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Min, &mask_vrf)
        .vector_final()
        .commit_trim::<m![Tf % 8]>()
        .commit()
}

fn row_sum(
    ctx: &mut Context,
    masked_exp: &DmTensor<f32, Chip, Cluster, Slice, m![Gf, Tf]>,
) -> DmTensor<f32, Chip, Cluster, Slice, m![Gf]> {
    ctx.main
        .begin(masked_exp.view())
        .fetch::<m![Gf, Tf / 8], m![Tf % 8]>()
        .collect::<m![Gf, Tf / 8], m![Tf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gf, Tf / 4], m![Tf % 4]>()
        .vector_intra_slice_reduce::<Tf, m![Gf], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .transpose::<m![Gf / 2], m![Gf % 2 # 8]>()
        .commit_trim::<m![Gf % 2]>()
        .commit()
}

fn attention_weighted_values(
    ctx: &mut Context,
    masked_exp: &DmTensor<f32, Chip, Cluster, Slice, m![Gf, Tf]>,
    v: &HbmTensor<bf16, Chip, m![Tf, Df]>,
) -> DmTensor<bf16, Chip, Cluster, PerQueryHead, m![Df]> {
    type QueryHeadAndRows = m![Gf, Tf / 32];

    let attention_weights: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Tf]> = ctx
        .main
        .begin(masked_exp.view())
        .fetch::<m![Gf, Tf / 8], m![Tf % 8]>()
        .collect::<m![Gf, Tf / 8], m![Tf % 8]>()
        .cast::<bf16, m![Tf % 8 # 16]>()
        .commit_trim::<m![Tf % 8]>()
        .commit();
    let attention_weights: DmTensor<bf16, Chip, Cluster, QueryHeadAndRows, m![Tf % 32]> =
        attention_weights.to_dm(&mut ctx.tdma);

    let v: DmTensor<bf16, Chip, Cluster, QueryHeadAndRows, m![Tf % 32, Df]> = v.to_dm(&mut ctx.tdma);
    let v: DmTensor<bf16, Chip, Cluster, QueryHeadAndRows, m![Df, Tf % 32]> = ctx
        .main
        .begin(v.view())
        .fetch::<m![Df / 8, Tf % 32], m![Df % 8 # 16]>()
        .collect::<m![Df / 8, Tf % 32], m![Df % 8 # 16]>()
        .transpose::<m![Df / 8, Tf / 4 % 8, Df % 8], m![Tf % 4 # 16]>()
        .commit_trim::<m![Tf % 4]>()
        .commit();
    let v_trf: TrfTensor<bf16, Chip, Cluster, QueryHeadAndRows, m![Df % 8], m![Df / 8, Tf % 32]> = ctx
        .sub
        .begin(v.view())
        .fetch::<m![Df % 8, Df / 8, Tf / 16 % 2], m![Tf % 16]>()
        .collect::<m![Df % 8, Df / 8, Tf / 16 % 2], m![Tf % 16]>()
        .to_trf();

    let weighted_sum: DmTensor<bf16, Chip, Cluster, QueryHeadAndRows, m![Df]> = ctx
        .main
        .begin(attention_weights.view())
        .fetch::<m![Tf / 16 % 2], m![Tf % 16]>()
        .collect::<m![Tf / 16 % 2], m![Tf % 16]>()
        .contract_outer::<m![Df / 8], m![Tf % 32], _, _, _>(&v_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Df / 8]>()
        .contract_lane::<m![Df / 8], m![Df % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit();
    ctx.main
        .begin(weighted_sum.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![Gf, 1 # 16], m![Df / 8]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}

pub(crate) fn attend_first_page(
    ctx: &mut Context,
    q: &HbmTensor<bf16, Chip, m![Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    mask: &HbmTensor<f32, Chip, m![Tf]>,
    running_max: &mut HbmTensor<f32, Chip, m![Gf]>,
    running_sum: &mut HbmTensor<f32, Chip, m![Gf]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![Gf, Df]>,
) {
    let qk = query_key_scores(ctx, q, k);
    let max = row_max(ctx, &qk);
    let max_vrf = per_head_to_vrf(ctx, &max);

    let exp = masked_exp(ctx, &qk, &max_vrf, mask);
    let sum = row_sum(ctx, &exp);
    let result = attention_weighted_values(ctx, &exp, v);

    max.view().to_hbm_view(&mut ctx.tdma, running_max.view_mut());
    sum.view().to_hbm_view(&mut ctx.tdma, running_sum.view_mut());
    result.view().to_hbm_view(&mut ctx.tdma, out_hbm.view_mut());
}

pub(crate) fn attend_next_page(
    ctx: &mut Context,
    q: &HbmTensor<bf16, Chip, m![Gf, Df]>,
    k: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    v: &HbmTensor<bf16, Chip, m![Tf, Df]>,
    mask: &HbmTensor<f32, Chip, m![Tf]>,
    running_max: &mut HbmTensor<f32, Chip, m![Gf]>,
    running_sum: &mut HbmTensor<f32, Chip, m![Gf]>,
    out_hbm: &mut HbmTensor<bf16, Chip, m![Gf, Df]>,
) {
    let qk = query_key_scores(ctx, q, k);
    let max = row_max(ctx, &qk);

    let old_max: DmTensor<f32, Chip, Cluster, Slice, m![Gf]> = running_max.to_dm(&mut ctx.tdma);
    let old_max_vrf = per_head_to_vrf(ctx, &old_max);

    let max: DmTensor<f32, Chip, Cluster, Slice, m![Gf]> = ctx
        .main
        .begin(max.view())
        .fetch::<m![Gf / 8], m![Gf % 8]>()
        .collect::<m![Gf / 8], m![Gf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Max, &old_max_vrf)
        .vector_final()
        .commit_trim::<m![Gf % 8]>()
        .commit();
    max.view().to_hbm_view(&mut ctx.tdma, running_max.view_mut());

    let rescale: DmTensor<f32, Chip, Cluster, Slice, m![Gf]> = ctx
        .main
        .begin(max.view())
        .fetch::<m![Gf / 8], m![Gf % 8]>()
        .collect::<m![Gf / 8], m![Gf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gf / 4], m![Gf % 4]>()
        .vector_fp_binary_with_mode(FpBinaryOp::SubF, BinaryArgMode::Mode10, &old_max_vrf)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![Gf / 8], m![Gf % 8]>()
        .vector_final()
        .commit_trim::<m![Gf % 8]>()
        .commit();
    let rescale_vrf = per_head_to_vrf(ctx, &rescale);
    let max_vrf = per_head_to_vrf(ctx, &max);

    let exp = masked_exp(ctx, &qk, &max_vrf, mask);
    let expsum = row_sum(ctx, &exp);
    let expsum_vrf = per_head_to_vrf(ctx, &expsum);

    let sum: DmTensor<f32, Chip, Cluster, Slice, m![Gf]> = running_sum.to_dm(&mut ctx.tdma);
    let sum: DmTensor<f32, Chip, Cluster, Slice, m![Gf]> = ctx
        .main
        .begin(sum.view())
        .fetch::<m![Gf / 8], m![Gf % 8]>()
        .collect::<m![Gf / 8], m![Gf % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gf / 4], m![Gf % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &rescale_vrf)
        .vector_widen_concat::<m![Gf / 8], m![Gf % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &expsum_vrf)
        .vector_final()
        .commit_trim::<m![Gf % 8]>()
        .commit();

    sum.view().to_hbm_view(&mut ctx.tdma, running_sum.view_mut());

    let result = attention_weighted_values(ctx, &exp, v);
    let result_vrf: VrfTensor<f32, Chip, Cluster, PerQueryHead, m![Df]> = ctx
        .sub
        .begin(result.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .to_vrf();

    let out: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> = out_hbm.to_dm(&mut ctx.tdma);
    let out: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> = ctx
        .main
        .begin(out.view())
        .fetch::<m![Gf, Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gf, Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gf, Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &rescale_vrf)
        .vector_widen_concat::<m![Gf, Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit();
    out.view().to_hbm_view(&mut ctx.tdma, out_hbm.view_mut());

    let out: DmTensor<bf16, Chip, Cluster, PerQueryHead, m![Df]> = out_hbm.to_dm(&mut ctx.tdma);
    let out: DmTensor<bf16, Chip, Cluster, PerQueryHead, m![Df]> = ctx
        .main
        .begin(out.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &result_vrf)
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit();

    out.view().to_hbm_view(&mut ctx.tdma, out_hbm.view_mut());
}

pub(crate) fn divide_by_softmax_sum<Slice: M>(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]>,
    running_sum: &HbmTensor<f32, Chip, m![Gf]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> {
    let sum: DmTensor<f32, Chip, Cluster, Slice, m![Gf]> = running_sum.to_dm(&mut ctx.tdma);
    let sum_vrf = per_head_to_vrf(ctx, &sum);

    ctx.main
        .begin(input.view())
        .fetch::<m![Gf, Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gf, Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gf, Df / 4], m![Df % 4]>()
        .vector_fp_div(&sum_vrf)
        .vector_widen_concat::<m![Gf, Df / 8], m![Df % 8]>()
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit()
}

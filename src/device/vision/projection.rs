
use furiosa_opt_std::prelude::*;

use crate::axes::{Dummy256, H, Mv, Ov, Rv};
use crate::device::layout::{Cluster, Replicated, Slice};
use crate::{Chip, EPS};

const OV_F32: f32 = Ov::SIZE as f32;

type EmbeddingRows = m![Mv / 120, 1 # 8];

fn broadcast_patch(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Rv]>,
) -> DmTensor<bf16, Chip, Cluster, Replicated, m![Rv]> {
    let x: DmTensor<bf16, Chip, Cluster, m![Dummy256], m![Rv]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![1], m![Rv]>()
        .switch::<m![Dummy256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![Rv / 16], m![Rv % 16]>()
        .commit_trim::<m![Rv % 16]>()
        .commit();

    unsafe { x.reshape() }
}

fn patch_partial(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Replicated, m![Rv]>,
    weight: &HbmTensor<bf16, Chip, m![Mv, Rv]>,
    offset: usize,
) -> DmTensor<bf16, Chip, Cluster, EmbeddingRows, m![Mv % 120]> {
    let x: DmTensorView<'_, bf16, Chip, Cluster, EmbeddingRows, m![Rv]> = unsafe { x.view().reshape() };
    let x_half = x.tile::<m![Rv], 768, m![Rv = 768 # 6912]>(offset);
    let x_trf: TrfTensor<bf16, Chip, Cluster, EmbeddingRows, m![1], m![Rv = 768]> = ctx
        .sub
        .begin(x_half)
        .fetch::<m![1], m![Rv = 768]>()
        .collect::<m![Rv = 768 / 16], m![Rv = 768 % 16]>()
        .to_trf();

    let weight_half = weight.view().tile::<m![Rv], 768, m![Mv, Rv = 768 # 6912]>(offset);
    let weight_dm: DmTensor<bf16, Chip, Cluster, EmbeddingRows, m![Mv % 120, Rv = 768]> =
        weight_half.to_dm(&mut ctx.tdma);

    ctx.main
        .begin(weight_dm.view())
        .fetch::<m![Mv % 120, Rv = 768 / 16], m![Rv = 768 % 16]>()
        .collect::<m![Mv % 120, Rv = 768 / 16], m![Rv = 768 % 16]>()
        .contract_outer::<m![Mv % 120, Rv = 768 / 32], m![Rv = 768 % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Mv % 120]>()
        .contract_lane::<m![Mv % 120], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Mv / 4 % 30], m![Mv % 4 # 16]>()
        .commit_trim::<m![Mv % 4]>()
        .commit()
}

fn add_partials_mv(
    ctx: &mut Context,
    a: DmTensor<bf16, Chip, Cluster, EmbeddingRows, m![Mv % 120]>,
    b: DmTensor<bf16, Chip, Cluster, EmbeddingRows, m![Mv % 120]>,
) -> DmTensor<bf16, Chip, Cluster, EmbeddingRows, m![Mv % 120]> {
    let a_vrf: VrfTensor<f32, Chip, Cluster, EmbeddingRows, m![Mv % 120]> = ctx
        .sub
        .begin(a.view())
        .fetch::<m![1], m![Mv % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![Mv / 8 % 15], m![Mv % 8]>()
        .to_vrf();

    ctx.main
        .begin(b.view())
        .fetch::<m![1], m![Mv % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![Mv / 8 % 15], m![Mv % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &a_vrf)
        .vector_final()
        .cast::<bf16, m![Mv % 8 # 16]>()
        .commit_trim::<m![Mv % 8]>()
        .commit()
}

pub(crate) fn patch_projection(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Rv]>,
    weight: &HbmTensor<bf16, Chip, m![Mv, Rv]>,
    bias: &HbmTensor<bf16, Chip, m![Mv]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Mv]> {
    let x = broadcast_patch(ctx, x);

    const CHUNK: usize = 768;

    let chunk0 = patch_partial(ctx, &x, weight, 0);
    let chunk1 = patch_partial(ctx, &x, weight, CHUNK);
    let chunk2 = patch_partial(ctx, &x, weight, 2 * CHUNK);
    let chunk3 = patch_partial(ctx, &x, weight, 3 * CHUNK);
    let chunk4 = patch_partial(ctx, &x, weight, 4 * CHUNK);
    let chunk5 = patch_partial(ctx, &x, weight, 5 * CHUNK);
    let chunk6 = patch_partial(ctx, &x, weight, 6 * CHUNK);
    let chunk7 = patch_partial(ctx, &x, weight, 7 * CHUNK);
    let chunk8 = patch_partial(ctx, &x, weight, 8 * CHUNK);

    let sum01 = add_partials_mv(ctx, chunk0, chunk1);
    let sum23 = add_partials_mv(ctx, chunk2, chunk3);
    let sum45 = add_partials_mv(ctx, chunk4, chunk5);
    let sum67 = add_partials_mv(ctx, chunk6, chunk7);

    let sum0123 = add_partials_mv(ctx, sum01, sum23);
    let sum4567 = add_partials_mv(ctx, sum45, sum67);
    let sum0_to_7 = add_partials_mv(ctx, sum0123, sum4567);
    let sum = add_partials_mv(ctx, sum0_to_7, chunk8);

    let bias: DmTensor<bf16, Chip, Cluster, EmbeddingRows, m![Mv % 120]> = bias.to_dm(&mut ctx.tdma);
    let bias_vrf: VrfTensor<f32, Chip, Cluster, EmbeddingRows, m![Mv % 120]> = ctx
        .sub
        .begin(bias.view())
        .fetch::<m![1], m![Mv % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![Mv / 8 % 15], m![Mv % 8]>()
        .to_vrf();

    let output: DmTensor<bf16, Chip, Cluster, EmbeddingRows, m![Mv % 120]> = ctx
        .main
        .begin(sum.view())
        .fetch::<m![1], m![Mv % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![Mv / 8 % 15], m![Mv % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &bias_vrf)
        .vector_final()
        .cast::<bf16, m![Mv % 8 # 16]>()
        .commit_trim::<m![Mv % 8]>()
        .commit();

    ctx.main
        .begin(output.view())
        .fetch::<m![Mv / 8 % 15], m![Mv % 8 # 16]>()
        .switch::<Slice, m![Mv / 8 % 15, Mv / 120]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 8 })
        .collect::<m![Mv / 8 % 15, Mv / 120], m![Mv % 8 # 16]>()
        .commit_trim::<m![Mv % 8]>()
        .commit()
}

fn mean_square_ov(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, Slice, m![Ov]>,
) -> DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> {
    ctx.main
        .begin(input.view())
        .fetch::<m![Ov / 16], m![Ov % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ov / 8], m![Ov % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ov / 4], m![Ov % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ov, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(OV_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

fn rms_ov(
    ctx: &mut Context,
    mean_square: &DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]>,
) -> DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> {
    ctx.main
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
        .commit()
}

fn rmsnorm_without_weight(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ov]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Ov]> {
    let mean_square = mean_square_ov(ctx, x);
    let rms = rms_ov(ctx, &mean_square);
    let rms_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(x.view())
        .fetch::<m![Ov / 16], m![Ov % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ov / 8], m![Ov % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ov / 4], m![Ov % 4]>()
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![Ov / 8], m![Ov % 8]>()
        .vector_final()
        .cast::<bf16, m![Ov % 8 # 16]>()
        .commit_trim::<m![Ov % 8]>()
        .commit()
}

fn broadcast_output(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ov]>,
) -> DmTensor<bf16, Chip, Cluster, Replicated, m![Ov]> {
    let x: DmTensor<bf16, Chip, Cluster, m![Dummy256], m![Ov]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![1], m![Ov]>()
        .switch::<m![Dummy256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![Ov / 16], m![Ov % 16]>()
        .commit_trim::<m![Ov % 16]>()
        .commit();

    unsafe { x.reshape() }
}

type HiddenRows = m![H / 120, 1 # 8];

fn project_partial(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Replicated, m![Ov]>,
    weight: &HbmTensor<bf16, Chip, m![H, Ov]>,
    offset: usize,
) -> DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]> {
    let x: DmTensorView<'_, bf16, Chip, Cluster, HiddenRows, m![Ov]> = unsafe { x.view().reshape() };
    let x_half = x.tile::<m![Ov], 768, m![Ov = 768 # 3840]>(offset);
    let x_trf: TrfTensor<bf16, Chip, Cluster, HiddenRows, m![1], m![Ov = 768]> = ctx
        .sub
        .begin(x_half)
        .fetch::<m![1], m![Ov = 768]>()
        .collect::<m![Ov = 768 / 16], m![Ov = 768 % 16]>()
        .to_trf();

    let weight_half = weight.view().tile::<m![Ov], 768, m![H, Ov = 768 # 3840]>(offset);
    let weight_dm: DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120, Ov = 768]> = weight_half.to_dm(&mut ctx.tdma);

    ctx.main
        .begin(weight_dm.view())
        .fetch::<m![H % 120, Ov = 768 / 16], m![Ov = 768 % 16]>()
        .collect::<m![H % 120, Ov = 768 / 16], m![Ov = 768 % 16]>()
        .contract_outer::<m![H % 120, Ov = 768 / 32], m![Ov = 768 % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120]>()
        .contract_lane::<m![H % 120], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H / 4 % 30], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit()
}

fn add_partials_h(
    ctx: &mut Context,
    a: DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]>,
    b: DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]>,
) -> DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]> {
    let a_vrf: VrfTensor<f32, Chip, Cluster, HiddenRows, m![H % 120]> = ctx
        .sub
        .begin(a.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();

    ctx.main
        .begin(b.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &a_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

pub(crate) fn project_to_text_hidden(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Ov]>,
    weight: &HbmTensor<bf16, Chip, m![H, Ov]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let x = rmsnorm_without_weight(ctx, x);
    let x = broadcast_output(ctx, &x);

    const CHUNK: usize = 768;

    let chunk0 = project_partial(ctx, &x, weight, 0);
    let chunk1 = project_partial(ctx, &x, weight, CHUNK);
    let chunk2 = project_partial(ctx, &x, weight, 2 * CHUNK);
    let chunk3 = project_partial(ctx, &x, weight, 3 * CHUNK);
    let chunk4 = project_partial(ctx, &x, weight, 4 * CHUNK);

    let sum01 = add_partials_h(ctx, chunk0, chunk1);
    let sum23 = add_partials_h(ctx, chunk2, chunk3);
    let sum0123 = add_partials_h(ctx, sum01, sum23);
    let sum = add_partials_h(ctx, sum0123, chunk4);

    ctx.main
        .begin(sum.view())
        .fetch::<m![H / 8 % 15], m![H % 8 # 16]>()
        .switch::<Slice, m![H / 8 % 15, H / 120]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 8 })
        .collect::<m![H / 8 % 15, H / 120], m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}


use furiosa_opt_std::prelude::*;

use crate::axes::{Mv, Rv};
use crate::{Chip, EPS};

const MV_F32: f32 = Mv::SIZE as f32;
const RV_F32: f32 = Rv::SIZE as f32;

fn mean<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Mv]>,
) -> DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> {
    ctx.main
        .begin(x.view())
        .fetch::<m![Mv / 16], m![Mv % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Mv / 8], m![Mv % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Mv / 4], m![Mv % 4]>()
        .vector_intra_slice_reduce::<Mv, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(MV_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

fn to_vrf<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]>,
) -> VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]> {
    ctx.sub
        .begin(x.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

pub(crate) fn normalize<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Mv]>,
    weight: &HbmTensor<bf16, Chip, m![Mv]>,
    bias: &HbmTensor<bf16, Chip, m![Mv]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Mv]> {
    let mean = mean(ctx, x);
    let mean_vrf = to_vrf(ctx, &mean);

    let centred: DmTensor<f32, Chip, Cluster, Slice, m![Mv]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Mv / 16], m![Mv % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Mv / 8], m![Mv % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Mv / 4], m![Mv % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &mean_vrf)
        .vector_widen_concat::<m![Mv / 8], m![Mv % 8]>()
        .vector_final()
        .commit_trim::<m![Mv % 8]>()
        .commit();

    let var: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .main
        .begin(centred.view())
        .fetch::<m![Mv / 8], m![Mv % 8]>()
        .collect::<m![Mv / 8], m![Mv % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Mv / 4], m![Mv % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Mv, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(MV_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let std_dev: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .main
        .begin(var.view())
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
    let std_vrf = to_vrf(ctx, &std_dev);

    let weight_dm: DmTensor<bf16, Chip, Cluster, Slice, m![Mv]> = weight.to_dm(&mut ctx.tdma);
    let bias_dm: DmTensor<bf16, Chip, Cluster, Slice, m![Mv]> = bias.to_dm(&mut ctx.tdma);

    const TILES: usize = Mv::SIZE / 480;

    let mut output: DmTensor<bf16, Chip, Cluster, Slice, m![Mv]> = DmTensor::new();

    for i in 0..TILES {
        let centred_tile = centred.view().tile::<m![Mv], 480, m![Mv = 480 # 3840]>(480 * i);
        let weight_tile = weight_dm.view().tile::<m![Mv], 480, m![Mv = 480 # 3840]>(480 * i);
        let bias_tile = bias_dm.view().tile::<m![Mv], 480, m![Mv = 480 # 3840]>(480 * i);

        let weight_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Mv = 480]> = ctx
            .sub
            .begin(weight_tile)
            .fetch::<m![1], m![Mv = 480]>()
            .fetch_cast::<f32>()
            .collect::<m![Mv = 480 / 8], m![Mv = 480 % 8]>()
            .to_vrf();

        let bias_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Mv = 480]> = ctx
            .sub
            .begin(bias_tile)
            .fetch::<m![1], m![Mv = 480]>()
            .fetch_cast::<f32>()
            .collect::<m![Mv = 480 / 8], m![Mv = 480 % 8]>()
            .to_vrf();

        ctx.main
            .begin(centred_tile)
            .fetch::<m![1], m![Mv = 480]>()
            .collect::<m![Mv = 480 / 8], m![Mv = 480 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Mv = 480 / 4], m![Mv = 480 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
            .vector_fp_div(&std_vrf)
            .vector_widen_concat::<m![Mv = 480 / 8], m![Mv = 480 % 8]>()
            .vector_clip(ClipBinaryOpF32::Add, &bias_vrf)
            .vector_final()
            .cast::<bf16, m![Mv = 480 % 8 # 16]>()
            .commit_trim::<m![Mv = 480 % 8]>()
            .commit_view(output.view_mut().tile::<m![Mv], 480, m![Mv = 480 #{!} 3840]>(480 * i));
    }

    output
}

fn mean_patch<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Rv]>,
) -> DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> {
    ctx.main
        .begin(x.view())
        .fetch::<m![Rv / 16], m![Rv % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Rv / 8], m![Rv % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rv / 4], m![Rv % 4]>()
        .vector_intra_slice_reduce::<Rv, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(RV_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

pub(crate) fn normalize_patch<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Rv]>,
    weight: &HbmTensor<bf16, Chip, m![Rv]>,
    bias: &HbmTensor<bf16, Chip, m![Rv]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Rv]> {
    let mean = mean_patch(ctx, x);
    let mean_vrf = to_vrf(ctx, &mean);

    let centred: DmTensor<f32, Chip, Cluster, Slice, m![Rv]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Rv / 16], m![Rv % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Rv / 8], m![Rv % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rv / 4], m![Rv % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &mean_vrf)
        .vector_widen_concat::<m![Rv / 8], m![Rv % 8]>()
        .vector_final()
        .commit_trim::<m![Rv % 8]>()
        .commit();

    let var: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .main
        .begin(centred.view())
        .fetch::<m![Rv / 8], m![Rv % 8]>()
        .collect::<m![Rv / 8], m![Rv % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rv / 4], m![Rv % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Rv, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(RV_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let std_dev: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .main
        .begin(var.view())
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
    let std_vrf = to_vrf(ctx, &std_dev);

    let weight_dm: DmTensor<bf16, Chip, Cluster, Slice, m![Rv]> = weight.to_dm(&mut ctx.tdma);
    let bias_dm: DmTensor<bf16, Chip, Cluster, Slice, m![Rv]> = bias.to_dm(&mut ctx.tdma);

    const TILES: usize = Rv::SIZE / 864;

    let mut output: DmTensor<bf16, Chip, Cluster, Slice, m![Rv]> = DmTensor::new();

    for i in 0..TILES {
        let centred_tile = centred.view().tile::<m![Rv], 864, m![Rv = 864 # 6912]>(864 * i);
        let weight_tile = weight_dm.view().tile::<m![Rv], 864, m![Rv = 864 # 6912]>(864 * i);
        let bias_tile = bias_dm.view().tile::<m![Rv], 864, m![Rv = 864 # 6912]>(864 * i);

        let weight_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Rv = 864]> = ctx
            .sub
            .begin(weight_tile)
            .fetch::<m![1], m![Rv = 864]>()
            .fetch_cast::<f32>()
            .collect::<m![Rv = 864 / 8], m![Rv = 864 % 8]>()
            .to_vrf();

        let bias_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Rv = 864]> = ctx
            .sub
            .begin(bias_tile)
            .fetch::<m![1], m![Rv = 864]>()
            .fetch_cast::<f32>()
            .collect::<m![Rv = 864 / 8], m![Rv = 864 % 8]>()
            .to_vrf();

        ctx.main
            .begin(centred_tile)
            .fetch::<m![1], m![Rv = 864]>()
            .collect::<m![Rv = 864 / 8], m![Rv = 864 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Rv = 864 / 4], m![Rv = 864 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
            .vector_fp_div(&std_vrf)
            .vector_widen_concat::<m![Rv = 864 / 8], m![Rv = 864 % 8]>()
            .vector_clip(ClipBinaryOpF32::Add, &bias_vrf)
            .vector_final()
            .cast::<bf16, m![Rv = 864 % 8 # 16]>()
            .commit_trim::<m![Rv = 864 % 8]>()
            .commit_view(output.view_mut().tile::<m![Rv], 864, m![Rv = 864 #{!} 6912]>(864 * i));
    }

    output
}

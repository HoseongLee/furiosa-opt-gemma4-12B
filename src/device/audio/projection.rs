
use furiosa_opt_std::prelude::*;

use crate::axes::{Aa, Dummy256, H};
use crate::device::layout::{Cluster, Replicated, Slice};
use crate::{Chip, EPS};

const AA_F32: f32 = Aa::SIZE as f32;
type HiddenRows = m![H / 120, 1 # 8];

fn mean_square(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, Slice, m![Aa]>,
) -> DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> {
    ctx.main
        .begin(input.view())
        .fetch::<m![Aa / 16], m![Aa % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Aa / 8], m![Aa % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Aa / 4], m![Aa % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Aa, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(AA_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit()
}

fn rms(
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
    input: &DmTensor<bf16, Chip, Cluster, Slice, m![Aa]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Aa]> {
    let mean_square = mean_square(ctx, input);
    let rms = rms(ctx, &mean_square);
    let rms_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(input.view())
        .fetch::<m![Aa / 16], m![Aa % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Aa / 8], m![Aa % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Aa / 4], m![Aa % 4]>()
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![Aa / 8], m![Aa % 8]>()
        .vector_final()
        .cast::<bf16, m![Aa % 8 # 16]>()
        .commit_trim::<m![Aa % 8]>()
        .commit()
}

fn broadcast_audio(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, Slice, m![Aa]>,
) -> DmTensor<bf16, Chip, Cluster, Replicated, m![Aa]> {
    let input: DmTensor<bf16, Chip, Cluster, m![Dummy256], m![Aa]> = ctx
        .main
        .begin(input.view())
        .fetch::<m![1], m![Aa]>()
        .switch::<m![Dummy256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![Aa / 16], m![Aa % 16]>()
        .commit_trim::<m![Aa % 16]>()
        .commit();
    unsafe { input.reshape() }
}

fn partial(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, Replicated, m![Aa]>,
    weight: &HbmTensor<bf16, Chip, m![H, Aa]>,
) -> DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120]> {
    let input: DmTensorView<'_, bf16, Chip, Cluster, HiddenRows, m![Aa]> = unsafe { input.view().reshape() };
    let input_trf: TrfTensor<bf16, Chip, Cluster, HiddenRows, m![1], m![Aa]> = ctx
        .sub
        .begin(input)
        .fetch::<m![1], m![Aa]>()
        .collect::<m![Aa / 16], m![Aa % 16]>()
        .to_trf();
    let weight: DmTensor<bf16, Chip, Cluster, HiddenRows, m![H % 120, Aa]> = weight.to_dm(&mut ctx.tdma);

    ctx.main
        .begin(weight.view())
        .fetch::<m![H % 120, Aa / 16], m![Aa % 16]>()
        .collect::<m![H % 120, Aa / 16], m![Aa % 16]>()
        .contract_outer::<m![H % 120, Aa / 32], m![Aa % 32], _, _, _>(&input_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120]>()
        .contract_lane::<m![H % 120], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H / 4 % 30], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit()
}

pub(crate) fn project_frame(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, Slice, m![Aa]>,
    weight: &HbmTensor<bf16, Chip, m![H, Aa]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let normalized = rmsnorm_without_weight(ctx, input);
    let input = broadcast_audio(ctx, &normalized);
    let p = partial(ctx, &input, weight);

    ctx.main
        .begin(p.view())
        .fetch::<m![H / 8 % 15], m![H % 8 # 16]>()
        .switch::<Slice, m![H / 8 % 15, H / 120]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 8 })
        .collect::<m![H / 8 % 15, H / 120], m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}


use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Ds, E, Gs, Ns};
use crate::device::layout::{Cluster, Slice};

type KvHeadsAcrossSlices = m![1 # 32, Ns];

pub(crate) fn apply_rope(
    ctx: &mut Context,
    q: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]>,
    k: &DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
) -> (
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]>,
) {
    let cos: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> = cos.dma_gather_scaled(rope_offset);
    let sin: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> = sin.dma_gather_scaled(rope_offset);

    let cos: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = cos.to_dm(&mut ctx.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = sin.to_dm(&mut ctx.tdma);

    let cos_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .sub
        .begin(cos.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let sin_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .sub
        .begin(sin.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let q: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8, Gs, Ds]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![Ns], m![Gs, Ds]>()
        .switch::<KvHeadsAcrossSlices, m![1 # 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![1 # 8, Gs, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();

    let k: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![1 # 8, Ds]> = ctx
        .main
        .begin(k.view())
        .fetch::<m![Ns], m![Ds]>()
        .switch::<KvHeadsAcrossSlices, m![1 # 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![1 # 8, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();

    let q: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![1], m![Gs, Ds]>()
        .collect::<m![Gs, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();

    let k: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .main
        .begin(k.view())
        .fetch::<m![1], m![Ds]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();

    let first_half_q = q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(0);
    let second_half_q = q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(128);

    let mut rotate_half_q: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = DmTensor::new();

    ctx.main
        .begin(first_half_q)
        .fetch::<m![Gs], m![Ds = 128]>()
        .collect::<m![Gs, Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotate_half_q
                .view_mut()
                .tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(128),
        );

    ctx.main
        .begin(second_half_q)
        .fetch::<m![Gs], m![Ds = 128]>()
        .collect::<m![Gs, Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotate_half_q
                .view_mut()
                .tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(0),
        );

    let first_half_k = k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(0);
    let second_half_k = k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128);

    let mut rotate_half_k: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = DmTensor::new();

    ctx.main
        .begin(first_half_k)
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128));

    ctx.main
        .begin(second_half_k)
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(0));

    let q_cos: DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let q_sin: DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = ctx
        .main
        .begin(rotate_half_q.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let q_sin_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = ctx
        .sub
        .begin(q_sin.view())
        .fetch::<m![Gs, Ds / 8], m![Ds % 8]>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let result_q: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Gs, Ds]> = ctx
        .main
        .begin(q_cos.view())
        .fetch::<m![Gs, Ds / 8], m![Ds % 8]>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &q_sin_vrf)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let k_cos: DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .main
        .begin(k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let k_sin: DmTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .main
        .begin(rotate_half_k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let k_sin_vrf: VrfTensor<f32, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .sub
        .begin(k_sin.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let result_k: DmTensor<bf16, Chip, Cluster, KvHeadsAcrossSlices, m![Ds]> = ctx
        .main
        .begin(k_cos.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &k_sin_vrf)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let result_q: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> = ctx
        .main
        .begin(result_q.view())
        .fetch::<m![1], m![Gs, Ds]>()
        .switch::<Slice, m![Ns]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })
        .collect::<m![Ns, Gs, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();

    let result_k: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Ds]> = ctx
        .main
        .begin(result_k.view())
        .fetch::<m![1], m![Ds]>()
        .switch::<Slice, m![Ns]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })
        .collect::<m![Ns, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit();

    (result_q, result_k)
}

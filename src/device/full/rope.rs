
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Df, E, Gf};
use crate::device::layout::{Cluster, Slice};

pub(crate) fn apply_rope(
    ctx: &mut Context,
    q: &DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]>,
    k: &DmTensor<bf16, Chip, Cluster, Slice, m![Df]>,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Df]>,
    sin: &HbmTensor<bf16, Chip, m![E, Df]>,
) -> (
    DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]>,
    DmTensor<bf16, Chip, Cluster, Slice, m![Df]>,
) {
    let cos: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = cos.dma_gather_scaled(rope_offset);
    let sin: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = sin.dma_gather_scaled(rope_offset);

    let cos: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = cos.to_dm(&mut ctx.tdma);
    let sin: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = sin.to_dm(&mut ctx.tdma);

    let cos_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Df]> = ctx
        .sub
        .begin(cos.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .to_vrf();

    let sin_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Df]> = ctx
        .sub
        .begin(sin.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .to_vrf();

    let first_half_q = q.view().tile::<m![Df], 256, m![Gf, Df = 256 # 512]>(0);
    let second_half_q = q.view().tile::<m![Df], 256, m![Gf, Df = 256 # 512]>(256);

    let mut rotate_half_q: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> = DmTensor::new();

    ctx.main
        .begin(first_half_q)
        .fetch::<m![Gf], m![Df = 256]>()
        .collect::<m![Gf, Df = 256 / 16], m![Df = 256 % 16]>()
        .commit_trim::<m![Df = 256 % 16]>()
        .commit_view(
            rotate_half_q
                .view_mut()
                .tile::<m![Df], 256, m![Gf, Df = 256 #{!} 512]>(256),
        );

    ctx.main
        .begin(second_half_q)
        .fetch::<m![Gf], m![Df = 256]>()
        .collect::<m![Gf, Df = 256 / 16], m![Df = 256 % 16]>()
        .commit_trim::<m![Df = 256 % 16]>()
        .commit_view(
            rotate_half_q
                .view_mut()
                .tile::<m![Df], 256, m![Gf, Df = 256 #{!} 512]>(0),
        );

    let first_half_k = k.view().tile::<m![Df], 256, m![Df = 256 # 512]>(0);
    let second_half_k = k.view().tile::<m![Df], 256, m![Df = 256 # 512]>(256);

    let mut rotate_half_k: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = DmTensor::new();

    ctx.main
        .begin(first_half_k)
        .fetch::<m![1], m![Df = 256]>()
        .collect::<m![Df = 256 / 16], m![Df = 256 % 16]>()
        .commit_trim::<m![Df = 256 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![Df], 256, m![Df = 256 #{!} 512]>(256));

    ctx.main
        .begin(second_half_k)
        .fetch::<m![1], m![Df = 256]>()
        .collect::<m![Df = 256 / 16], m![Df = 256 % 16]>()
        .commit_trim::<m![Df = 256 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![Df], 256, m![Df = 256 #{!} 512]>(0));

    let mut result_q: DmTensor<bf16, Chip, Cluster, Slice, m![Gf, Df]> = DmTensor::new();

    for g in 0..Gf::SIZE {
        let q_cos: DmTensor<f32, Chip, Cluster, Slice, m![Df]> = ctx
            .main
            .begin(q.view().tile::<m![Gf], 1, m![Gf = 1 # 16, Df]>(g))
            .fetch::<m![Df / 16], m![Df % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Df / 8], m![Df % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Df / 4], m![Df % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
            .vector_widen_concat::<m![Df / 8], m![Df % 8]>()
            .vector_final()
            .commit_trim::<m![Df % 8]>()
            .commit();

        let q_sin: DmTensor<f32, Chip, Cluster, Slice, m![Df]> = ctx
            .main
            .begin(rotate_half_q.view().tile::<m![Gf], 1, m![Gf = 1 # 16, Df]>(g))
            .fetch::<m![Df / 16], m![Df % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![Df / 8], m![Df % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Df / 4], m![Df % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
            .vector_widen_concat::<m![Df / 8], m![Df % 8]>()
            .vector_final()
            .commit_trim::<m![Df % 8]>()
            .commit();

        let q_sin_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Df]> = ctx
            .sub
            .begin(q_sin.view())
            .fetch::<m![Df / 8], m![Df % 8]>()
            .collect::<m![Df / 8], m![Df % 8]>()
            .to_vrf();

        ctx.main
            .begin(q_cos.view())
            .fetch::<m![Df / 8], m![Df % 8]>()
            .collect::<m![Df / 8], m![Df % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, &q_sin_vrf)
            .vector_final()
            .cast::<bf16, m![Df % 8 # 16]>()
            .commit_trim::<m![Df % 8]>()
            .commit_view(result_q.view_mut().tile::<m![Gf], 1, m![Gf = 1 #{!} 16, Df]>(g));
    }

    let k_cos: DmTensor<f32, Chip, Cluster, Slice, m![Df]> = ctx
        .main
        .begin(k.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Df / 8], m![Df % 8]>()
        .vector_final()
        .commit_trim::<m![Df % 8]>()
        .commit();

    let k_sin: DmTensor<f32, Chip, Cluster, Slice, m![Df]> = ctx
        .main
        .begin(rotate_half_k.view())
        .fetch::<m![Df / 16], m![Df % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Df / 4], m![Df % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Df / 8], m![Df % 8]>()
        .vector_final()
        .commit_trim::<m![Df % 8]>()
        .commit();

    let k_sin_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Df]> = ctx
        .sub
        .begin(k_sin.view())
        .fetch::<m![Df / 8], m![Df % 8]>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .to_vrf();

    let result_k: DmTensor<bf16, Chip, Cluster, Slice, m![Df]> = ctx
        .main
        .begin(k_cos.view())
        .fetch::<m![Df / 8], m![Df % 8]>()
        .collect::<m![Df / 8], m![Df % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &k_sin_vrf)
        .vector_final()
        .cast::<bf16, m![Df % 8 # 16]>()
        .commit_trim::<m![Df % 8]>()
        .commit();

    (result_q, result_k)
}

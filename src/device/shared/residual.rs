
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, Mv};
use crate::device::layout::{Cluster, Slice};

pub(crate) fn add(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    residual: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    const TILES: usize = H::SIZE / 480;

    let mut output: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = DmTensor::new();

    for i in 0..TILES {
        let x_tile = x.view().tile::<m![H], 480, m![H = 480 # 3840]>(480 * i);
        let residual_tile = residual.view().tile::<m![H], 480, m![H = 480 # 3840]>(480 * i);

        let residual_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![H = 480]> = ctx
            .sub
            .begin(residual_tile)
            .fetch::<m![1], m![H = 480]>()
            .fetch_cast::<f32>()
            .collect::<m![H = 480 / 8], m![H = 480 % 8]>()
            .to_vrf();

        ctx.main
            .begin(x_tile)
            .fetch::<m![1], m![H = 480]>()
            .fetch_cast::<f32>()
            .collect::<m![H = 480 / 8], m![H = 480 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
            .vector_final()
            .cast::<bf16, m![H = 480 % 8 # 16]>()
            .commit_trim::<m![H = 480 % 8]>()
            .commit_view(output.view_mut().tile::<m![H], 480, m![H = 480 #{!} 3840]>(480 * i));
    }

    output
}

pub(crate) fn scale_by_layer_gate(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let scalar: DmTensor<bf16, Chip, Cluster, Slice, m![1 # 8]> = scalar.to_dm(&mut ctx.tdma);
    let scalar: VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .sub
        .begin(scalar.view())
        .fetch::<m![1], m![1 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(input.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scalar)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

pub(crate) fn add_vision<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Mv]>,
    residual: &DmTensor<bf16, Chip, Cluster, Slice, m![Mv]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![Mv]> {
    const TILES: usize = Mv::SIZE / 480;

    let mut output: DmTensor<bf16, Chip, Cluster, Slice, m![Mv]> = DmTensor::new();

    for i in 0..TILES {
        let x_tile = x.view().tile::<m![Mv], 480, m![Mv = 480 # 3840]>(480 * i);
        let residual_tile = residual.view().tile::<m![Mv], 480, m![Mv = 480 # 3840]>(480 * i);

        let residual_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Mv = 480]> = ctx
            .sub
            .begin(residual_tile)
            .fetch::<m![1], m![Mv = 480]>()
            .fetch_cast::<f32>()
            .collect::<m![Mv = 480 / 8], m![Mv = 480 % 8]>()
            .to_vrf();

        ctx.main
            .begin(x_tile)
            .fetch::<m![1], m![Mv = 480]>()
            .fetch_cast::<f32>()
            .collect::<m![Mv = 480 / 8], m![Mv = 480 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
            .vector_final()
            .cast::<bf16, m![Mv = 480 % 8 # 16]>()
            .commit_trim::<m![Mv = 480 % 8]>()
            .commit_view(output.view_mut().tile::<m![Mv], 480, m![Mv = 480 #{!} 3840]>(480 * i));
    }

    output
}

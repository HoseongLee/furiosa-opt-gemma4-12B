
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, L};
use crate::device::layout::{Cluster, Replicated, Slice};

const INVSQRT2: f32 = 0.70710678118f32;

pub(crate) type UpGateRows = m![L / 60];
pub(crate) type UpGateRowsPaired = m![L / 120, 1 # 2];

pub(crate) fn project_up_and_gate(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, Cluster, UpGateRows, m![1], m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> (
    DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
) {
    const ROWS_PER_SLICE: usize = 60;
    const ROWS_PER_PASS: usize = 4;
    const PASSES: usize = ROWS_PER_SLICE / ROWS_PER_PASS;

    let mut up: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]> = DmTensor::new();

    let up_weight_packed: DmTensor<f4e2m1, Chip, Cluster, UpGateRows, m![L % 60, H]> =
        up_weight_packed.to_dm(&mut ctx.tdma);
    let up_weight_packed: DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H]> = ctx
        .main
        .begin(up_weight_packed.view())
        .fetch::<m![L % 60], m![H]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![L % 60, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();

    let up_weight_scale: DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H / 16]> =
        up_weight_scale.to_dm(&mut ctx.tdma);

    for i in 0..PASSES {
        let up_weight_scale_vrf: VrfTensor<f32, Chip, Cluster, UpGateRows, m![L % 60 = 4, H / 16]> = ctx
            .sub
            .begin(
                up_weight_scale
                    .view()
                    .tile::<m![L % 60], 4, m![L % 60 = 4 # 60, H / 16]>(4 * i),
            )
            .fetch::<m![L % 60 = 4], m![H / 16]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 4, H / 16 / 8], m![H / 16 % 8]>()
            .to_vrf();

        let up_weight: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60 = 4, H]> = ctx
            .main
            .begin(
                up_weight_packed
                    .view()
                    .tile::<m![L % 60], 4, m![L % 60 = 4 # 60, H]>(4 * i),
            )
            .fetch::<m![L % 60 = 4, H / 32], m![H % 32]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 4, H / 8], m![H % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L % 60 = 4, H / 4], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up_weight_scale_vrf)
            .vector_widen_concat::<m![L % 60 = 4, H / 8], m![H % 8]>()
            .vector_final()
            .cast::<bf16, m![H % 8 # 16]>()
            .commit_trim::<m![H % 8]>()
            .commit();

        ctx.main
            .begin(up_weight.view())
            .fetch::<m![L % 60 = 4, H / 16], m![H % 16]>()
            .collect::<m![L % 60 = 4, H / 16], m![H % 16]>()
            .contract_outer::<m![L % 60 = 4, H / 32], m![H % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![L % 60 = 4]>()
            .contract_lane::<m![L % 60 = 4], m![1 # 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![1], m![L % 60 = 4 # 16]>()
            .commit_trim::<m![L % 60 = 4]>()
            .commit_view(up.view_mut().tile::<m![L % 60], 4, m![L % 60 = 4 #{!} 60]>(4 * i));
    }

    let mut gate: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]> = DmTensor::new();

    let gate_weight_packed: DmTensor<f4e2m1, Chip, Cluster, UpGateRows, m![L % 60, H]> =
        gate_weight_packed.to_dm(&mut ctx.tdma);
    let gate_weight_packed: DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H]> = ctx
        .main
        .begin(gate_weight_packed.view())
        .fetch::<m![L % 60], m![H]>()
        .fetch_table_lookup::<f8e4m3>()
        .collect::<m![L % 60, H / 32], m![H % 32]>()
        .commit_trim::<m![H % 32]>()
        .commit();

    let gate_weight_scale: DmTensor<f8e4m3, Chip, Cluster, UpGateRows, m![L % 60, H / 16]> =
        gate_weight_scale.to_dm(&mut ctx.tdma);

    for i in 0..PASSES {
        let gate_weight_scale_vrf: VrfTensor<f32, Chip, Cluster, UpGateRows, m![L % 60 = 4, H / 16]> = ctx
            .sub
            .begin(
                gate_weight_scale
                    .view()
                    .tile::<m![L % 60], 4, m![L % 60 = 4 # 60, H / 16]>(4 * i),
            )
            .fetch::<m![L % 60 = 4], m![H / 16]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 4, H / 16 / 8], m![H / 16 % 8]>()
            .to_vrf();

        let gate_weight: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60 = 4, H]> = ctx
            .main
            .begin(
                gate_weight_packed
                    .view()
                    .tile::<m![L % 60], 4, m![L % 60 = 4 # 60, H]>(4 * i),
            )
            .fetch::<m![L % 60 = 4, H / 32], m![H % 32]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 4, H / 8], m![H % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L % 60 = 4, H / 4], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate_weight_scale_vrf)
            .vector_widen_concat::<m![L % 60 = 4, H / 8], m![H % 8]>()
            .vector_final()
            .cast::<bf16, m![H % 8 # 16]>()
            .commit_trim::<m![H % 8]>()
            .commit();

        ctx.main
            .begin(gate_weight.view())
            .fetch::<m![L % 60 = 4, H / 16], m![H % 16]>()
            .collect::<m![L % 60 = 4, H / 16], m![H % 16]>()
            .contract_outer::<m![L % 60 = 4, H / 32], m![H % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![L % 60 = 4]>()
            .contract_lane::<m![L % 60 = 4], m![1 # 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![1], m![L % 60 = 4 # 16]>()
            .commit_trim::<m![L % 60 = 4]>()
            .commit_view(gate.view_mut().tile::<m![L % 60], 4, m![L % 60 = 4 #{!} 60]>(4 * i));
    }

    (up, gate)
}

pub(crate) fn feedforward(
    ctx: &mut Context,
    x: DmTensor<bf16, Chip, Cluster, Replicated, m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let x: DmTensor<bf16, Chip, Cluster, UpGateRows, m![H]> = unsafe { x.reshape() };
    let x_trf: TrfTensor<bf16, Chip, Cluster, UpGateRows, m![1], m![H]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let (up, gate) = project_up_and_gate(
        ctx,
        &x_trf,
        up_weight_packed,
        gate_weight_packed,
        up_weight_scale,
        gate_weight_scale,
    );
    let x = geglu(ctx, up, gate, up_global_scale, gate_global_scale);
    let x = x.to_dm(&mut ctx.tdma);
    let down = project_down(ctx, &x, down_weight_packed, down_weight_scale);

    let down_global_scale: DmTensor<f32, Chip, Cluster, UpGateRowsPaired, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, UpGateRowsPaired, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = ctx
        .main
        .begin(down.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    down
}

pub(crate) fn geglu(
    ctx: &mut Context,
    up: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    gate: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> {
    let up: DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = ctx
        .main
        .begin(up.view())
        .fetch::<m![L / 4 % 15], m![L % 4 # 16]>()
        .switch::<UpGateRowsPaired, m![L / 4 % 15, L / 60 % 2]>(SwitchConfig::Broadcast1 { slice1: 2, slice0: 1 })
        .collect::<m![L / 4 % 15, L / 60 % 2], m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit();

    let up_global_scale: DmTensor<f32, Chip, Cluster, UpGateRowsPaired, m![1 # 8]> =
        up_global_scale.to_dm(&mut ctx.tdma);
    let up_global_scale_vrf: VrfTensor<f32, Chip, Cluster, UpGateRowsPaired, m![1 # 8]> = ctx
        .sub
        .begin(up_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let up: DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = ctx
        .main
        .begin(up.view())
        .fetch::<m![1], m![L % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up_global_scale_vrf)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gate_global_scale: DmTensor<f32, Chip, Cluster, UpGateRowsPaired, m![1 # 8]> =
        gate_global_scale.to_dm(&mut ctx.tdma);
    let gate_global_scale_vrf: VrfTensor<f32, Chip, Cluster, UpGateRowsPaired, m![1 # 8]> = ctx
        .sub
        .begin(gate_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let gate: DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = ctx
        .main
        .begin(gate.view())
        .fetch::<m![L / 4 % 15], m![L % 4 # 16]>()
        .switch::<UpGateRowsPaired, m![L / 4 % 15, L / 60 % 2]>(SwitchConfig::Broadcast1 { slice1: 2, slice0: 1 })
        .collect::<m![L / 4 % 15, L / 60 % 2], m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit();

    let gate: DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = ctx
        .main
        .begin(gate.view())
        .fetch::<m![1], m![L % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate_global_scale_vrf)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gelu: DmTensor<f32, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = ctx
        .sub
        .begin(gate.view())
        .fetch::<m![1], m![L % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INVSQRT2)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gelu_vrf: VrfTensor<f32, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = ctx
        .sub
        .begin(gelu.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .to_vrf();

    ctx.main
        .begin(up.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu_vrf)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit()
}

pub(crate) type DownRows = m![H / 120, 1 # 8];
pub(crate) type DownRowsByColumns = m![H / 120, L / 1920];

pub(crate) fn project_down(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, DownRows, m![L]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let x: DmTensor<bf16, Chip, Cluster, DownRowsByColumns, m![L % 1920]> = x.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<bf16, Chip, Cluster, DownRowsByColumns, m![1], m![L % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![L / 16 % 120], m![L % 16]>()
        .collect::<m![L / 16 % 120], m![L % 16]>()
        .to_trf();

    const ROWS_PER_SLICE: usize = 120;
    const ROWS_PER_PASS: usize = 4;
    const PASSES: usize = ROWS_PER_SLICE / ROWS_PER_PASS;
    const HALVES_PER_PASS: usize = 2;

    let mut down: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120]> = DmTensor::new();

    let down_weight_scale: DmTensor<f8e4m3, Chip, Cluster, DownRows, m![H % 120, L / 16]> =
        down_weight_scale.to_dm(&mut ctx.tdma);

    for i in 0..PASSES {
        let mut down_weight: DmTensor<bf16, Chip, Cluster, DownRowsByColumns, m![H % 120 = 4, L % 1920]> =
            DmTensor::new();

        for j in 0..HALVES_PER_PASS {
            let down_weight_packed: DmTensor<f4e2m1, Chip, Cluster, DownRows, m![H % 120 = 2, L]> = down_weight_packed
                .view()
                .tile::<m![H % 120], 2, m![H / 120, H % 120 = 2 # 120, L]>(4 * i + 2 * j)
                .to_dm(&mut ctx.tdma);
            let down_weight_packed: DmTensor<f8e4m3, Chip, Cluster, DownRows, m![H % 120 = 2, L]> = ctx
                .main
                .begin(down_weight_packed.view())
                .fetch::<m![H % 120 = 2], m![L]>()
                .fetch_table_lookup::<f8e4m3>()
                .collect::<m![H % 120 = 2, L / 32], m![L % 32]>()
                .commit_trim::<m![L % 32]>()
                .commit();

            let down_weight_scale_vrf: VrfTensor<f32, Chip, Cluster, DownRows, m![H % 120 = 2, L / 16]> = ctx
                .sub
                .begin(
                    down_weight_scale
                        .view()
                        .tile::<m![H % 120], 2, m![H % 120 = 2 # 120, L / 16]>(4 * i + 2 * j),
                )
                .fetch::<m![H % 120 = 2], m![L / 16]>()
                .fetch_cast::<f32>()
                .collect::<m![H % 120 = 2, L / 16 / 8], m![L / 16 % 8]>()
                .to_vrf();

            let down_weight_tile: DmTensor<bf16, Chip, Cluster, DownRows, m![H % 120 = 2, L]> = ctx
                .main
                .begin(down_weight_packed.view())
                .fetch::<m![H % 120 = 2, L / 32], m![L % 32]>()
                .fetch_cast::<f32>()
                .collect::<m![H % 120 = 2, L / 8], m![L % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![H % 120 = 2, L / 4], m![L % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_weight_scale_vrf)
                .vector_widen_concat::<m![H % 120 = 2, L / 8], m![L % 8]>()
                .vector_final()
                .cast::<bf16, m![L % 8 # 16]>()
                .commit_trim::<m![L % 8]>()
                .commit();
            let down_weight_tile: DmTensor<bf16, Chip, Cluster, DownRowsByColumns, m![H % 120 = 2, L % 1920]> =
                down_weight_tile.to_dm(&mut ctx.tdma);

            down_weight_tile.view().to_dm_view(
                &mut ctx.tdma,
                down_weight
                    .view_mut()
                    .tile::<m![H % 120 = 4], 2, m![H % 120 = 4 = 2 #{!} 4, L % 1920]>(2 * j),
            );
        }

        ctx.main
            .begin(down_weight.view())
            .fetch::<m![H % 120 = 4, L / 16 % 120], m![L % 16]>()
            .collect::<m![H % 120 = 4, L / 16 % 120], m![L % 16]>()
            .contract_outer::<m![H % 120 = 4, L / 32 % 60], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![H % 120 = 4]>()
            .contract_lane::<m![H % 120 = 4], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<DownRows, m![H % 120 = 4]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![1], m![H % 120 = 4 # 16]>()
            .commit_trim::<m![H % 120 = 4]>()
            .commit_view(down.view_mut().tile::<m![H % 120], 4, m![H % 120 = 4 #{!} 120]>(4 * i));
    }

    down.to_dm(&mut ctx.tdma)
}


use furiosa_opt_std::prelude::*;

use crate::axes::{Ds, Gs, Ns, Ts};
use crate::device::layout::{Cluster, Slice};
use crate::{Chip, EPS};

pub(crate) fn attend(
    ctx: &mut Context,
    q: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    k: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    v: &HbmTensor<bf16, Chip, m![Ts, Ns, Ds]>,
    mask: &HbmTensor<f32, Chip, m![Ts]>,
    out: &mut HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
) {
    type KvRowsAcrossSlices = m![Ts / 8, 1 # 2];
    type KvHeadAndRows = m![Ns, Ts / 32];

    let q: DmTensor<bf16, Chip, Cluster, KvRowsAcrossSlices, m![Ns, Gs, Ds]> = q.to_dm(&mut ctx.tdma);
    let k: DmTensor<bf16, Chip, Cluster, KvRowsAcrossSlices, m![Ts % 8, Ns, Ds]> = k.to_dm(&mut ctx.tdma);
    let k_trf: TrfTensor<bf16, Chip, Cluster, KvRowsAcrossSlices, m![Ts % 8], m![Ns, Ds]> = ctx
        .sub
        .begin(k.view())
        .fetch::<m![Ts % 8, Ns, Ds / 16], m![Ds % 16]>()
        .collect::<m![Ts % 8, Ns, Ds / 16], m![Ds % 16]>()
        .to_trf();

    let qk: DmTensor<bf16, Chip, Cluster, KvRowsAcrossSlices, m![Ns, Gs, Ts % 8]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![Ns, Gs, Ds / 16], m![Ds % 16]>()
        .collect::<m![Ns, Gs, Ds / 16], m![Ds % 16]>()
        .contract_outer::<m![Ns, Gs, Ds / 32], m![Ds % 32], _, _, _>(&k_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ns, Gs]>()
        .contract_lane::<m![Ns, Gs], m![Ts % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Ts % 8 # 16]>()
        .commit_trim::<m![Ts % 8]>()
        .commit();

    let qk: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ts]> = ctx
        .main
        .begin(qk.view())
        .fetch::<m![Ns, Gs], m![Ts % 8 # 16]>()
        .switch::<Slice, m![Ns, Gs, Ts / 8]>(SwitchConfig::Broadcast1 { slice1: 128, slice0: 2 })
        .collect::<m![Ns, Gs, Ts / 8], m![Ts % 8 # 16]>()
        .commit_trim::<m![Ts % 8]>()
        .commit();

    let max: DmTensor<f32, Chip, Cluster, Slice, m![Ns, Gs]> = ctx
        .main
        .begin(qk.view())
        .fetch::<m![Ns, Gs, Ts / 16], m![Ts % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns, Gs, Ts / 8], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Gs, Ts / 4], m![Ts % 4]>()
        .vector_intra_slice_reduce::<Ts, m![Ns, Gs], m![1 # 4]>(IntraSliceReduceOpF32::Max)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![Ns, Gs / 2], m![Gs % 2 # 8]>()
        .commit_trim::<m![Gs % 2]>()
        .commit();
    let max_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Ns, Gs]> = ctx
        .sub
        .begin(max.view())
        .fetch::<m![Ns / 4], m![Ns % 4, Gs]>()
        .collect::<m![Ns / 4], m![Ns % 4, Gs]>()
        .to_vrf();

    let exp: DmTensor<f32, Chip, Cluster, Slice, m![Ns, Gs, Ts]> = ctx
        .main
        .begin(qk.view())
        .fetch::<m![Ns, Gs, Ts / 16], m![Ts % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns, Gs, Ts / 8], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Gs, Ts / 4], m![Ts % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &max_vrf)
        .vector_fp_unary(FpUnaryOp::Exp)
        .vector_widen_concat::<m![Ns, Gs, Ts / 8], m![Ts % 8]>()
        .vector_final()
        .commit_trim::<m![Ts % 8]>()
        .commit();

    let mask: DmTensor<f32, Chip, Cluster, Slice, m![Ts]> = mask.to_dm(&mut ctx.tdma);
    let mask_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Ts]> = ctx
        .sub
        .begin(mask.view())
        .fetch::<m![Ts / 8], m![Ts % 8]>()
        .collect::<m![Ts / 8], m![Ts % 8]>()
        .to_vrf();

    let exp: DmTensor<f32, Chip, Cluster, Slice, m![Ns, Gs, Ts]> = ctx
        .main
        .begin(exp.view())
        .fetch::<m![Ns, Gs, Ts / 8], m![Ts % 8]>()
        .collect::<m![Ns, Gs, Ts / 8], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Min, &mask_vrf)
        .vector_final()
        .commit_trim::<m![Ts % 8]>()
        .commit();

    let sum: DmTensor<f32, Chip, Cluster, Slice, m![Ns, Gs]> = ctx
        .main
        .begin(exp.view())
        .fetch::<m![Ns, Gs, Ts / 8], m![Ts % 8]>()
        .collect::<m![Ns, Gs, Ts / 8], m![Ts % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Gs, Ts / 4], m![Ts % 4]>()
        .vector_intra_slice_reduce::<Ts, m![Ns, Gs], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .transpose::<m![Ns, Gs / 2], m![Gs % 2 # 8]>()
        .commit_trim::<m![Gs % 2]>()
        .commit();
    let sum_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Ns, Gs]> = ctx
        .sub
        .begin(sum.view())
        .fetch::<m![Ns / 4], m![Ns % 4, Gs]>()
        .collect::<m![Ns / 4], m![Ns % 4, Gs]>()
        .to_vrf();

    let attention_weights: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ts]> = ctx
        .main
        .begin(exp.view())
        .fetch::<m![Ns, Gs, Ts / 8], m![Ts % 8]>()
        .collect::<m![Ns, Gs, Ts / 8], m![Ts % 8]>()
        .cast::<bf16, m![Ts % 8 # 16]>()
        .commit_trim::<m![Ts % 8]>()
        .commit();
    let attention_weights: DmTensor<bf16, Chip, Cluster, KvHeadAndRows, m![Gs, Ts % 32]> =
        attention_weights.to_dm(&mut ctx.tdma);

    let v: DmTensor<bf16, Chip, Cluster, KvHeadAndRows, m![Ts % 32, Ds]> = v.to_dm(&mut ctx.tdma);
    let v: DmTensor<bf16, Chip, Cluster, KvHeadAndRows, m![Ds, Ts % 32]> = ctx
        .main
        .begin(v.view())
        .fetch::<m![Ds / 8, Ts % 32], m![Ds % 8 # 16]>()
        .collect::<m![Ds / 8, Ts % 32], m![Ds % 8 # 16]>()
        .transpose::<m![Ds / 8, Ts / 4 % 8, Ds % 8], m![Ts % 4 # 16]>()
        .commit_trim::<m![Ts % 4]>()
        .commit();
    let v_trf: TrfTensor<bf16, Chip, Cluster, KvHeadAndRows, m![Ds % 8], m![Ds / 8, Ts % 32]> = ctx
        .sub
        .begin(v.view())
        .fetch::<m![Ds % 8, Ds / 8, Ts / 16 % 2], m![Ts % 16]>()
        .collect::<m![Ds % 8, Ds / 8, Ts / 16 % 2], m![Ts % 16]>()
        .to_trf();

    let contraction: DmTensor<bf16, Chip, Cluster, KvHeadAndRows, m![Gs, Ds]> = ctx
        .main
        .begin(attention_weights.view())
        .fetch::<m![Gs, Ts / 16 % 2], m![Ts % 16]>()
        .collect::<m![Gs, Ts / 16 % 2], m![Ts % 16]>()
        .contract_outer::<m![Gs, Ds / 8], m![Ts % 32], _, _, _>(&v_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Gs, Ds / 8]>()
        .contract_lane::<m![Gs, Ds / 8], m![Ds % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let combined: DmTensor<bf16, Chip, Cluster, m![Ns, 1 # 32], m![Gs, Ds]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![Ns, 1 # 32], m![Gs, Ds / 8]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();
    let combined: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> = combined.to_dm(&mut ctx.tdma);

    let output: DmTensor<bf16, Chip, Cluster, Slice, m![Ns, Gs, Ds]> = ctx
        .main
        .begin(combined.view())
        .fetch::<m![Ns, Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ns, Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ns, Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_div(&sum_vrf)
        .vector_widen_concat::<m![Ns, Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    output.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
}

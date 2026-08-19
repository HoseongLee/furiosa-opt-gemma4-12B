
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, W};
use crate::device::layout::Slice;

pub(crate) type Cluster = m![W / 8192 % 2];
pub(crate) type VocabRows = m![W / 32 % 256];

pub(crate) type LogitSlices = m![W / 512 % 16, W / 16384];
pub(crate) type LogitsPerSlice = m![W % 512];

pub(crate) fn logits(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    weight: &HbmTensor<bf16, Chip, m![W, H]>,
) -> DmTensor<bf16, Chip, Cluster, LogitSlices, LogitsPerSlice> {
    let mut logits: DmTensor<bf16, Chip, Cluster, VocabRows, m![W / 16384, W % 32]> = DmTensor::new();

    let x: DmTensor<bf16, Chip, Cluster, VocabRows, m![H]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .switch::<VocabRows, m![H / 16]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    let x_trf: TrfTensor<bf16, Chip, Cluster, VocabRows, m![1], m![H]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    const BANDS: usize = W::SIZE / 16384;

    for i in 0..BANDS {
        let weight_dm: DmTensor<bf16, Chip, Cluster, VocabRows, m![W % 32, H]> = weight
            .view()
            .tile::<m![W / 16384], 1, m![1 # 16, W % 16384, H]>(i)
            .to_dm(&mut ctx.tdma);

        ctx.main
            .begin(weight_dm.view())
            .fetch::<m![W % 32, H / 16], m![H % 16]>()
            .collect::<m![W % 32, H / 16], m![H % 16]>()
            .contract_outer::<m![W % 32, H / 32], m![H % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![W % 32]>()
            .contract_lane::<m![W % 32], m![1 # 8]>(LaneMode::Interleaved)
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![W / 4 % 8], m![W % 4 # 16]>()
            .commit_trim::<m![W % 4]>()
            .commit_view(logits.view_mut().tile::<m![W / 16384], 1, m![1 #{!} 16, W % 32]>(i));
    }

    ctx.main
        .begin(logits.view())
        .fetch::<m![W / 16384], m![W % 32]>()
        .switch::<LogitSlices, m![W / 32 % 16]>(SwitchConfig::InterTranspose {
            slice1: 16,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![W / 16 % 32], m![W % 16]>()
        .commit_trim::<m![W % 16]>()
        .commit()
}

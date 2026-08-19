
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;

pub(crate) type Cluster = m![1 # 2];
pub(crate) type Slice = m![1 # 256];

pub(crate) type Replicated = m![Dummy256];

pub(crate) fn broadcast_hidden(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Replicated, m![H]> {
    let x: DmTensor<bf16, Chip, Cluster, m![Dummy256], m![H]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![1], m![H]>()
        .switch::<m![Dummy256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();

    unsafe { x.reshape() }
}

pub(crate) fn broadcast_sliding_heads(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Qs]>,
) -> DmTensor<bf16, Chip, Cluster, Replicated, m![Qs]> {
    let x: DmTensor<bf16, Chip, Cluster, m![Dummy256], m![Qs]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![1], m![Qs]>()
        .switch::<m![Dummy256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![Qs / 16], m![Qs % 16]>()
        .commit_trim::<m![Qs % 16]>()
        .commit();

    unsafe { x.reshape() }
}

pub(crate) fn broadcast_full_heads(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Qf]>,
) -> DmTensor<bf16, Chip, Cluster, Replicated, m![Qf]> {
    let x: DmTensor<bf16, Chip, Cluster, m![Dummy256], m![Qf]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![1], m![Qf]>()
        .switch::<m![Dummy256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![Qf / 16], m![Qf % 16]>()
        .commit_trim::<m![Qf % 16]>()
        .commit();

    unsafe { x.reshape() }
}

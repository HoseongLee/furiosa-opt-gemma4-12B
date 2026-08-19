
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;
use crate::device::layout::{Cluster, Slice};
use crate::device::{shared, vision};

#[device(chip = 1)]
pub fn patch_embed(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![Rv]>,
    ln1_weight: &HbmTensor<bf16, Chip, m![Rv]>,
    ln1_bias: &HbmTensor<bf16, Chip, m![Rv]>,
    dense_weight: &HbmTensor<bf16, Chip, m![Mv, Rv]>,
    dense_bias: &HbmTensor<bf16, Chip, m![Mv]>,
    ln2_weight: &HbmTensor<bf16, Chip, m![Mv]>,
    ln2_bias: &HbmTensor<bf16, Chip, m![Mv]>,
    out: &mut HbmTensor<bf16, Chip, m![Mv]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Rv]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Rv]> =
        vision::layernorm::normalize_patch(ctx, &x, ln1_weight, ln1_bias);

    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Mv]> =
        vision::projection::patch_projection(ctx, &x, dense_weight, dense_bias);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Mv]> = vision::layernorm::normalize(ctx, &x, ln2_weight, ln2_bias);

    x.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
}

#[device(chip = 1)]
pub fn add_position_and_norm(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![Mv]>,
    pos_embed: &HbmTensor<bf16, Chip, m![Mv]>,
    ln3_weight: &HbmTensor<bf16, Chip, m![Mv]>,
    ln3_bias: &HbmTensor<bf16, Chip, m![Mv]>,
    out: &mut HbmTensor<bf16, Chip, m![Mv]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Mv]> = x.to_dm(&mut ctx.tdma);
    let pos_embed: DmTensor<bf16, Chip, Cluster, Slice, m![Mv]> = pos_embed.to_dm(&mut ctx.tdma);

    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Mv]> = shared::residual::add_vision(ctx, &x, &pos_embed);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Mv]> = vision::layernorm::normalize(ctx, &x, ln3_weight, ln3_bias);

    x.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
}

#[device(chip = 1)]
pub fn project_to_text_embedding(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![Ov]>,
    proj_weight: &HbmTensor<bf16, Chip, m![H, Ov]>,
    out: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Ov]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![H]> =
        vision::projection::project_to_text_hidden(ctx, &x, proj_weight);
    x.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
}

#[device(chip = 1)]
pub fn layernorm(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![Mv]>,
    weight: &HbmTensor<bf16, Chip, m![Mv]>,
    bias: &HbmTensor<bf16, Chip, m![Mv]>,
    out: &mut HbmTensor<bf16, Chip, m![Mv]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Mv]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<bf16, Chip, Cluster, Slice, m![Mv]> = vision::layernorm::normalize(ctx, &x, weight, bias);
    x.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
}

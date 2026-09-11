use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Aa, H};
use crate::device::audio;
use crate::device::layout::{Cluster, Slice};

#[device(chip = 1)]
pub fn audio_project_frame(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![Aa]>,
    weight: &HbmTensor<bf16, Chip, m![H, Aa]>,
    out: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let input: DmTensor<bf16, Chip, Cluster, Slice, m![Aa]> = input.to_dm(&mut device.tdma);
    let output = audio::projection::project_frame(device, &input, weight);
    output.view().to_hbm_view(&mut device.tdma, out.view_mut());
}

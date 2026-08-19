
use furiosa_opt_std::prelude::*;

use crate::axes::Rv;

const PATCH_SIZE: u32 = 16;
const POOLING_KERNEL_SIZE: u32 = 3;
const MODEL_PATCH_SIZE: u32 = PATCH_SIZE * POOLING_KERNEL_SIZE;
const MAX_SOFT_TOKENS: usize = 280;

pub struct ImagePatches {
    pub pixels: Vec<Vec<bf16>>,
    pub positions: Vec<(usize, usize)>,
}

const MAX_SOURCE_SIDE: u32 = 16384;
const MAX_DECODE_BYTES: u64 = 256 * 1024 * 1024;

pub fn load_from_bytes(bytes: &[u8]) -> Result<ImagePatches, Box<dyn std::error::Error>> {
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_SOURCE_SIDE);
    limits.max_image_height = Some(MAX_SOURCE_SIDE);
    limits.max_alloc = Some(MAX_DECODE_BYTES);

    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
    reader.limits(limits);
    patchify(reader.decode()?.to_rgb8())
}

fn aspect_preserving_size(height: u32, width: u32) -> Result<(u32, u32), String> {
    let max_raw_patches = MAX_SOFT_TOKENS as f64 * (POOLING_KERNEL_SIZE as f64).powi(2);
    let target_px = max_raw_patches * f64::from(PATCH_SIZE).powi(2);
    let factor = (target_px / (f64::from(height) * f64::from(width))).sqrt();

    let block = f64::from(MODEL_PATCH_SIZE);
    let mut target_height = (f64::from(height) * factor / block).floor() * block;
    let mut target_width = (f64::from(width) * factor / block).floor() * block;

    if target_height == 0.0 && target_width == 0.0 {
        return Err(format!(
            "cannot resize {width}x{height} to a multiple of {MODEL_PATCH_SIZE}"
        ));
    }
    let max_side = MAX_SOFT_TOKENS as f64 * block;
    if target_height == 0.0 {
        target_height = block;
        target_width = ((f64::from(width) / f64::from(height)).floor() * block).min(max_side);
    } else if target_width == 0.0 {
        target_width = block;
        target_height = ((f64::from(height) / f64::from(width)).floor() * block).min(max_side);
    }
    if target_height * target_width > target_px {
        return Err(format!(
            "resizing {width}x{height} to {target_width}x{target_height} exceeds {MAX_SOFT_TOKENS} patches"
        ));
    }
    Ok((target_height as u32, target_width as u32))
}

fn patchify(image: image::RgbImage) -> Result<ImagePatches, Box<dyn std::error::Error>> {
    let (width, height) = image.dimensions();
    let (target_height, target_width) = aspect_preserving_size(height, width)?;

    let resized = if (target_width, target_height) == (width, height) {
        image
    } else {
        image::imageops::resize(
            &image,
            target_width,
            target_height,
            image::imageops::FilterType::CatmullRom,
        )
    };

    let patch_cols = (target_width / MODEL_PATCH_SIZE) as usize;
    let patch_rows = (target_height / MODEL_PATCH_SIZE) as usize;
    let block_px = MODEL_PATCH_SIZE as usize;
    let stride = target_width as usize * 3;
    let raw = resized.as_raw();

    let mut pixels = Vec::with_capacity(patch_rows * patch_cols);
    let mut positions = Vec::with_capacity(patch_rows * patch_cols);
    for row in 0..patch_rows {
        for col in 0..patch_cols {
            let mut patch = Vec::with_capacity(Rv::SIZE);
            for local_row in 0..block_px {
                let y = row * block_px + local_row;
                let row_start = y * stride + col * block_px * 3;
                let row_bytes = &raw[row_start..row_start + block_px * 3];
                patch.extend(
                    row_bytes
                        .iter()
                        .map(|&channel| bf16::from_f32(f32::from(channel) / 255.0)),
                );
            }
            assert_eq!(patch.len(), Rv::SIZE);
            pixels.push(patch);
            positions.push((col, row));
        }
    }

    Ok(ImagePatches { pixels, positions })
}

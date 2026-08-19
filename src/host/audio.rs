
use furiosa_opt_std::prelude::bf16;

pub const SAMPLE_RATE: u32 = 16_000;
pub const SAMPLES_PER_TOKEN: usize = 640;
pub const MAX_SECONDS: u32 = 30;
pub const MAX_SAMPLES: usize = SAMPLE_RATE as usize * MAX_SECONDS as usize;

#[derive(Clone)]
pub struct AudioFrames {
    pub frames: Vec<Vec<bf16>>,
}

impl AudioFrames {
    pub fn len(&self) -> usize {
        self.frames.len()
    }
}

pub fn load_from_bytes(bytes: &[u8], format: &str) -> Result<AudioFrames, Box<dyn std::error::Error>> {
    if !format.eq_ignore_ascii_case("wav") {
        return Err(format!("audio format {format:?} is not supported; use wav").into());
    }
    let (sample_rate, samples) = decode_wav(bytes)?;
    let samples = resample_mono(&samples, sample_rate, SAMPLE_RATE)?;
    Ok(frame(samples))
}

fn read_u16(bytes: &[u8], at: usize) -> Result<u16, Box<dyn std::error::Error>> {
    let end = at.checked_add(2).ok_or("WAV offset overflow")?;
    Ok(u16::from_le_bytes(
        bytes.get(at..end).ok_or("truncated WAV")?.try_into()?,
    ))
}

fn read_u32(bytes: &[u8], at: usize) -> Result<u32, Box<dyn std::error::Error>> {
    let end = at.checked_add(4).ok_or("WAV offset overflow")?;
    Ok(u32::from_le_bytes(
        bytes.get(at..end).ok_or("truncated WAV")?.try_into()?,
    ))
}

fn decode_wav(bytes: &[u8]) -> Result<(u32, Vec<f32>), Box<dyn std::error::Error>> {
    if bytes.get(0..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WAVE") {
        return Err("audio is not a RIFF/WAVE file".into());
    }

    let mut at = 12usize;
    let mut format = None;
    let mut data = None;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let size = read_u32(bytes, at + 4)? as usize;
        let start = at + 8;
        let end = start.checked_add(size).ok_or("WAV chunk overflow")?;
        let chunk = bytes.get(start..end).ok_or("truncated WAV chunk")?;
        match id {
            b"fmt " => {
                if chunk.len() < 16 {
                    return Err("WAV fmt chunk is truncated".into());
                }
                format = Some((
                    read_u16(chunk, 0)?,
                    read_u16(chunk, 2)?,
                    read_u32(chunk, 4)?,
                    read_u16(chunk, 14)?,
                ));
            }
            b"data" => data = Some(chunk),
            _ => {}
        }
        at = end + (size & 1);
    }

    let (audio_format, channels, sample_rate, bits) = format.ok_or("WAV has no fmt chunk")?;
    let data = data.ok_or("WAV has no data chunk")?;
    if channels == 0 || sample_rate == 0 {
        return Err("WAV has invalid channel count or sample rate".into());
    }
    if !matches!(audio_format, 1 | 3) {
        return Err("WAV must contain PCM integer or IEEE float samples".into());
    }
    let bytes_per_sample = match (audio_format, bits) {
        (1, 8 | 16 | 24 | 32) => (bits / 8) as usize,
        (3, 32) => 4,
        _ => return Err(format!("unsupported WAV sample format: format={audio_format}, bits={bits}").into()),
    };
    let frame_bytes = bytes_per_sample * channels as usize;
    if frame_bytes == 0 || data.len() % frame_bytes != 0 {
        return Err("WAV data is not aligned to complete samples".into());
    }

    let mut mono = Vec::with_capacity(data.len() / frame_bytes);
    for frame in data.chunks_exact(frame_bytes) {
        let mut sum = 0.0f32;
        for channel in 0..channels as usize {
            let sample = &frame[channel * bytes_per_sample..(channel + 1) * bytes_per_sample];
            sum += match (audio_format, bits) {
                (1, 8) => (sample[0] as f32 - 128.0) / 128.0,
                (1, 16) => i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32768.0,
                (1, 24) => {
                    let value = ((sample[0] as i32) | ((sample[1] as i32) << 8) | ((sample[2] as i32) << 16)) << 8 >> 8;
                    value as f32 / 8_388_608.0
                }
                (1, 32) => i32::from_le_bytes(sample.try_into()?) as f32 / 2_147_483_648.0,
                (3, 32) => f32::from_le_bytes(sample.try_into()?),
                _ => unreachable!(),
            };
        }
        mono.push(sum / channels as f32);
    }
    Ok((sample_rate, mono))
}

fn resample_mono(samples: &[f32], from: u32, to: u32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if samples.is_empty() {
        return Err("audio contains no samples".into());
    }
    if samples.len() > MAX_SAMPLES.saturating_mul(from as usize) / SAMPLE_RATE as usize + from as usize {
        return Err("audio exceeds the 30 second limit".into());
    }
    if from == to {
        return Ok(samples.to_vec());
    }
    let output_len = ((samples.len() as u64 * to as u64) + from as u64 - 1) / from as u64;
    let mut output = Vec::with_capacity(output_len as usize);
    for i in 0..output_len as usize {
        let position = i as f64 * from as f64 / to as f64;
        let left = position.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let fraction = position - left as f64;
        output.push(samples[left] * (1.0 - fraction as f32) + samples[right] * fraction as f32);
    }
    if output.len() > MAX_SAMPLES {
        return Err("audio exceeds the 30 second limit".into());
    }
    Ok(output)
}

fn frame(samples: Vec<f32>) -> AudioFrames {
    let count = samples.len().div_ceil(SAMPLES_PER_TOKEN);
    let mut frames = Vec::with_capacity(count);
    for chunk in samples.chunks(SAMPLES_PER_TOKEN) {
        let mut frame = vec![bf16::from_f32(0.0); SAMPLES_PER_TOKEN];
        for (dst, &sample) in frame.iter_mut().zip(chunk) {
            *dst = bf16::from_f32(sample);
        }
        frames.push(frame);
    }
    AudioFrames { frames }
}

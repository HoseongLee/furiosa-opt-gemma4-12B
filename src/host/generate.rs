
use crate::Chip;
use crate::axes::{E, W};
use crate::host::audio::AudioFrames;
use crate::host::image::ImagePatches;
use crate::host::load::Model;
use crate::host::runtime::{self, Workspace};
use crate::host::sampling::{SamplingConfig, sample};
use crate::host::tokenizer::Tokenizer;
use furiosa_opt_std::prelude::*;
use llm_tokenizer::DecodeStream;
use std::collections::HashSet;

pub const EOS_TOKEN: usize = 1;
pub const END_OF_TURN_TOKEN: usize = 106;
pub const TOOL_RESPONSE_TOKEN: usize = 50;

pub const EOS_TOKENS: [usize; 3] = [EOS_TOKEN, END_OF_TURN_TOKEN, TOOL_RESPONSE_TOKEN];

pub const CHANNEL_OPEN_TOKEN: usize = 100;
pub const CHANNEL_CLOSE_TOKEN: usize = 101;

const CHANNEL_OPEN_TEXT: &str = "<|channel>";
const CHANNEL_CLOSE_TEXT: &str = "<channel|>";

pub const DEFAULT_MAX_NEW_TOKENS: usize = 300;
pub const BOI_TOKEN: usize = 255999;
pub const IMAGE_TOKEN: usize = 258880;
pub const EOI_TOKEN: usize = 258882;
pub const BOA_TOKEN: usize = 256000;
pub const AUDIO_TOKEN: usize = 258881;
pub const EOA_TOKEN: usize = 258883;

const SUPPRESSED_TOKENS: [usize; 2] = [EOI_TOKEN, EOA_TOKEN];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Channel {
    Answer,
    Name,
    Reasoning,
}

#[derive(Clone, Copy, Debug)]
pub enum Delta<'a> {
    Content(&'a str),
    Reasoning(&'a str),
}

#[derive(Clone, Copy)]
pub enum SoftToken {
    Image(usize),
    Audio(usize),
}

pub fn splice_multimodal_tokens(
    ids: Vec<usize>,
    image: Option<&ImagePatches>,
    audio: Option<&AudioFrames>,
) -> Result<(Vec<usize>, Vec<Option<SoftToken>>), String> {
    let image_markers = ids.iter().filter(|&&id| id == IMAGE_TOKEN).count();
    let audio_markers = ids.iter().filter(|&&id| id == AUDIO_TOKEN).count();
    if image.is_some() && image_markers != 1 {
        return Err(format!("expected one image placeholder, found {image_markers}"));
    }
    if audio.is_some() && audio_markers != 1 {
        return Err(format!("expected one audio placeholder, found {audio_markers}"));
    }

    let image_len = image.map_or(0, |value| value.pixels.len());
    let audio_len = audio.map_or(0, AudioFrames::len);
    let mut out_ids = Vec::with_capacity(ids.len() + image_len + audio_len + 4);
    let mut soft_at = Vec::with_capacity(out_ids.capacity());
    for id in ids {
        match id {
            IMAGE_TOKEN if image.is_some() => {
                out_ids.push(BOI_TOKEN);
                soft_at.push(None);
                for index in 0..image_len {
                    out_ids.push(IMAGE_TOKEN);
                    soft_at.push(Some(SoftToken::Image(index)));
                }
                out_ids.push(EOI_TOKEN);
                soft_at.push(None);
            }
            AUDIO_TOKEN if audio.is_some() => {
                out_ids.push(BOA_TOKEN);
                soft_at.push(None);
                for index in 0..audio_len {
                    out_ids.push(AUDIO_TOKEN);
                    soft_at.push(Some(SoftToken::Audio(index)));
                }
                out_ids.push(EOA_TOKEN);
                soft_at.push(None);
            }
            _ => {
                out_ids.push(id);
                soft_at.push(None);
            }
        }
    }
    Ok((out_ids, soft_at))
}

fn pending_stop_overlap(text: &str, stop_strings: &[&str]) -> usize {
    let mut hold = 0usize;
    for &stop in stop_strings {
        let max_len = (stop.len() - 1).min(text.len());
        for len in (1..=max_len).rev() {
            if stop.is_char_boundary(len) && text.ends_with(&stop[..len]) {
                hold = hold.max(len);
                break;
            }
        }
    }
    hold
}

fn step_over_marker(
    stream: &mut DecodeStream,
    id: usize,
    marker: &str,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let Some(delta) = stream.step(id as u32)? else {
        return Ok(None);
    };
    let held = delta.strip_suffix(marker).unwrap_or(&delta);
    Ok((!held.is_empty()).then(|| held.to_owned()))
}

pub async fn read_logits(ctx: &mut Context, logits: &HbmTensor<bf16, Chip, m![W]>) -> Vec<f32> {
    let logits: HostTensor<bf16, m![W]> = logits.to_host(&mut ctx.pdma).await;
    logits.into_vec().into_iter().map(bf16::to_f32).collect()
}

async fn sample_next(
    ctx: &mut Context,
    workspace: &Workspace,
    sampling: &SamplingConfig,
    banned: &HashSet<usize>,
    rng: &mut impl rand::Rng,
) -> usize {
    sample(&read_logits(ctx, &workspace.logits).await, sampling, banned, rng)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

pub struct GenerationRequest {
    pub prompt_ids: Vec<usize>,
    pub soft_at: Vec<Option<SoftToken>>,
    pub image: Option<ImagePatches>,
    pub audio: Option<AudioFrames>,
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
    pub stop_strings: Vec<String>,
    pub separate_reasoning: bool,
}

pub struct GenerationOutput {
    pub text: String,
    pub reasoning: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: FinishReason,
}

pub async fn generate(
    ctx: &mut Context,
    model: &Model,
    tokenizer: &Tokenizer,
    workspace: &mut Workspace,
    req: GenerationRequest,
    mut on_delta: impl FnMut(Delta) -> bool,
) -> Result<GenerationOutput, Box<dyn std::error::Error>> {
    if req.prompt_ids.is_empty() {
        return Err("prompt produced no tokens".into());
    }
    if req.prompt_ids.len().saturating_add(req.max_new_tokens) > E::SIZE {
        return Err(format!("prompt plus generation exceeds {} positions", E::SIZE).into());
    }
    if req.prompt_ids.len() != req.soft_at.len() {
        return Err(format!(
            "prompt_ids ({}) and soft_at ({}) must be parallel",
            req.prompt_ids.len(),
            req.soft_at.len()
        )
        .into());
    }

    let mut decode = workspace.begin_decode();
    for (pos, (&token, soft_token)) in req.prompt_ids.iter().zip(req.soft_at.iter()).enumerate() {
        match soft_token {
            Some(SoftToken::Image(i)) => {
                let patches = req.image.as_ref().expect("soft_at only set when an image was loaded");
                let embedding =
                    runtime::encode_vision_patch(ctx, model, &patches.pixels[*i], patches.positions[*i]).await;
                decode.run_with_embedding(ctx, workspace, &embedding, model, pos).await;
            }
            Some(SoftToken::Audio(i)) => {
                let audio = req.audio.as_ref().expect("soft_at only set when audio was loaded");
                let embedding = runtime::encode_audio_frame(ctx, model, &audio.frames[*i]).await;
                decode.run_with_embedding(ctx, workspace, &embedding, model, pos).await;
            }
            None => {
                decode.run(ctx, workspace, token, model, pos).await;
            }
        }
    }

    let mut rng = rand::rng();
    let banned: HashSet<usize> = SUPPRESSED_TOKENS.into_iter().collect();
    let mut next = sample_next(ctx, workspace, &req.sampling, &banned, &mut rng).await;

    let stop_strings: Vec<&str> = req
        .stop_strings
        .iter()
        .map(String::as_str)
        .filter(|s| !s.is_empty())
        .collect();
    let prompt_ids_u32: Vec<u32> = req.prompt_ids.iter().map(|&id| id as u32).collect();
    let mut stream = tokenizer.decode_stream(&prompt_ids_u32);
    let mut generated_count = 0usize;
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut emitted = 0usize;
    let mut stopped_on_stop_string = false;
    let mut cancelled = false;
    let mut channel = Channel::Answer;

    while !EOS_TOKENS.contains(&next) && generated_count < req.max_new_tokens {
        let (delta, switch_to) = match next {
            CHANNEL_OPEN_TOKEN if req.separate_reasoning => (
                step_over_marker(&mut stream, next, CHANNEL_OPEN_TEXT)?,
                Some(Channel::Name),
            ),
            CHANNEL_CLOSE_TOKEN if req.separate_reasoning => (
                step_over_marker(&mut stream, next, CHANNEL_CLOSE_TEXT)?,
                Some(Channel::Answer),
            ),
            _ => (stream.step(next as u32)?, None),
        };

        if let Some(delta) = delta {
            match channel {
                Channel::Name => {
                    if let Some((_, rest)) = delta.split_once('\n') {
                        channel = Channel::Reasoning;
                        if !rest.is_empty() {
                            reasoning.push_str(rest);
                            cancelled |= !on_delta(Delta::Reasoning(rest));
                        }
                    }
                }
                Channel::Reasoning => {
                    reasoning.push_str(&delta);
                    cancelled |= !on_delta(Delta::Reasoning(&delta));
                }
                Channel::Answer => {
                    text.push_str(&delta);
                    let full_match = stop_strings.iter().filter_map(|s| text.find(s)).min();
                    match full_match {
                        Some(cut) => {
                            if cut > emitted {
                                let _ = on_delta(Delta::Content(&text[emitted..cut]));
                            }
                            text.truncate(cut);
                            stopped_on_stop_string = true;
                            break;
                        }
                        None => {
                            let safe_len = text.len() - pending_stop_overlap(&text, &stop_strings);
                            if safe_len > emitted {
                                cancelled |= !on_delta(Delta::Content(&text[emitted..safe_len]));
                                emitted = safe_len;
                            }
                        }
                    }
                }
            }
        }
        if let Some(next_channel) = switch_to {
            channel = next_channel;
        }
        if cancelled {
            break;
        }

        generated_count += 1;
        if generated_count == req.max_new_tokens {
            break;
        }
        let pos = req.prompt_ids.len() + generated_count - 1;
        decode.run(ctx, workspace, next, model, pos).await;
        next = sample_next(ctx, workspace, &req.sampling, &banned, &mut rng).await;
    }
    if !stopped_on_stop_string && emitted < text.len() {
        let _ = on_delta(Delta::Content(&text[emitted..]));
    }

    let finish_reason = if req.max_new_tokens == 0 {
        FinishReason::Length
    } else if stopped_on_stop_string || EOS_TOKENS.contains(&next) {
        FinishReason::Stop
    } else {
        FinishReason::Length
    };

    Ok(GenerationOutput {
        text,
        reasoning,
        prompt_tokens: req.prompt_ids.len(),
        completion_tokens: generated_count,
        finish_reason,
    })
}

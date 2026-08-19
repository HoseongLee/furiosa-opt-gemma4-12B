
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

use furiosa_opt_std::prelude::AxisName;
use rand::Rng;
use serde::Serialize;

use super::schema::{
    AudioSpeechRequest, ChatChoice, ChatCompletionChunk, ChatCompletionMessage, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, ChunkChoice, ChunkDelta, CompletionChoice, CompletionChunk, CompletionRequest,
    CompletionResponse, ContentPart, ErrorResponse, MODEL_ID, MessageContent, ModelObject, ModelsList, OneOrMany,
    Usage,
};
use super::server::{self, AppState, HttpRequest, HttpResponse};
use super::worker::{Job, StreamEvent};
use crate::axes::{E, W};
use crate::host::audio::{self, AudioFrames};
use crate::host::generate::{self, FinishReason, GenerationRequest};
use crate::host::image::{self, ImagePatches};
use crate::host::sampling::SamplingConfig;

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn gen_id(prefix: &str) -> String {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let mut rng = rand::rng();
    let suffix: String = (0..24)
        .map(|_| ALPHABET[rng.random_range(0..ALPHABET.len())] as char)
        .collect();
    format!("{prefix}-{suffix}")
}

fn finish_reason_str(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

pub(crate) fn json<T: Serialize>(status: u16, body: &T) -> HttpResponse {
    HttpResponse::json(status, reason_phrase(status), body)
}

pub(crate) fn error(status: u16, error_type: &'static str, message: impl Into<String>) -> HttpResponse {
    json(status, &ErrorResponse::new(error_type, message))
}

fn error_param(status: u16, error_type: &'static str, message: impl Into<String>, param: &'static str) -> HttpResponse {
    json(status, &ErrorResponse::with_param(error_type, message, param))
}

fn validate_n(n: Option<u32>) -> Result<(), HttpResponse> {
    match n {
        Some(n) if n != 1 => Err(error_param(400, "invalid_request_error", "only n=1 is supported", "n")),
        _ => Ok(()),
    }
}

const MAX_STOP_STRINGS: usize = 8;
const MAX_STOP_STRING_BYTES: usize = 256;

fn validate_stop(stop: &[String]) -> Result<(), HttpResponse> {
    if stop.len() > MAX_STOP_STRINGS {
        return Err(error_param(
            400,
            "invalid_request_error",
            format!("at most {MAX_STOP_STRINGS} stop strings are supported"),
            "stop",
        ));
    }
    if let Some(long) = stop.iter().find(|s| s.len() > MAX_STOP_STRING_BYTES) {
        return Err(error_param(
            400,
            "invalid_request_error",
            format!(
                "a stop string may be at most {MAX_STOP_STRING_BYTES} bytes; got one of {}",
                long.len()
            ),
            "stop",
        ));
    }
    Ok(())
}

fn validate_unsupported(requested: Option<&'static str>) -> Result<(), HttpResponse> {
    match requested {
        Some(field) => Err(error_param(
            400,
            "invalid_request_error",
            format!("{field} is not supported by this server"),
            field,
        )),
        None => Ok(()),
    }
}

fn validate_top_k(top_k: Option<usize>) -> Result<(), HttpResponse> {
    match top_k {
        Some(k) if k > W::SIZE => Err(error_param(
            400,
            "invalid_request_error",
            format!("top_k must be at most the vocabulary size ({})", W::SIZE),
            "top_k",
        )),
        _ => Ok(()),
    }
}

fn validate_budget(prompt_len: usize, max_new_tokens: usize, param: &'static str) -> Result<(), HttpResponse> {
    if prompt_len.saturating_add(max_new_tokens) > E::SIZE {
        return Err(error_param(
            400,
            "invalid_request_error",
            format!(
                "prompt ({prompt_len} tokens) plus {max_new_tokens} generated tokens exceeds the {} position context",
                E::SIZE
            ),
            param,
        ));
    }
    Ok(())
}

fn submit_job(state: &AppState, request: GenerationRequest, stream: bool) -> mpsc::Receiver<StreamEvent> {
    let (tx, rx) = mpsc::channel::<StreamEvent>();
    state.worker.submit(Job {
        request,
        stream,
        events: tx,
    });
    rx
}

fn block_for_result(
    rx: mpsc::Receiver<StreamEvent>,
    on_done: impl FnOnce(generate::GenerationOutput) -> HttpResponse,
) -> HttpResponse {
    match rx.recv() {
        Ok(StreamEvent::Done(output)) => on_done(output),
        Ok(StreamEvent::Error(message)) => error(500, "internal_error", message),
        Ok(StreamEvent::Delta(_) | StreamEvent::ReasoningDelta(_)) | Err(_) => {
            error(500, "internal_error", "generation worker closed unexpectedly")
        }
    }
}

fn validate_model(model: &str) -> Result<(), HttpResponse> {
    if model != MODEL_ID {
        return Err(error_param(
            404,
            "invalid_request_error",
            format!("model {model:?} not found; this server only serves {MODEL_ID:?}"),
            "model",
        ));
    }
    Ok(())
}

pub(crate) fn not_found() -> HttpResponse {
    error(404, "invalid_request_error", "unknown endpoint")
}

fn audio_not_implemented() -> HttpResponse {
    error(
        501,
        "not_implemented",
        "audio output is not implemented; audio input is supported only in chat completions",
    )
}

fn constant_time_eq(provided: &[u8], expected: &[u8]) -> bool {
    let mut difference = (provided.len() ^ expected.len()) as u8;
    for (index, byte) in provided.iter().enumerate() {
        difference |= byte ^ expected.get(index).copied().unwrap_or(0);
    }
    difference == 0
}

pub(crate) fn check_auth(req: &HttpRequest, state: &AppState) -> Option<HttpResponse> {
    let Some(expected) = &state.api_key else { return None };
    let provided = req.header("authorization").and_then(|value| {
        let (scheme, token) = value.split_once(' ')?;
        scheme.eq_ignore_ascii_case("bearer").then(|| token.trim())
    });
    let ok = provided.is_some_and(|provided| constant_time_eq(provided.as_bytes(), expected.as_bytes()));
    if ok {
        None
    } else {
        Some(error(401, "invalid_request_error", "invalid or missing API key"))
    }
}

pub(crate) fn list_models(state: &AppState) -> HttpResponse {
    if !state.worker.is_alive() {
        return error(
            503,
            "internal_error",
            "the generation worker is not running; this process needs to be restarted",
        );
    }
    json(
        200,
        &ModelsList {
            object: "list",
            data: vec![ModelObject {
                id: MODEL_ID,
                object: "model",
                created: now_secs(),
                owned_by: "furiosa",
            }],
        },
    )
}

const BASE64_ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    fn value(byte: u8) -> Option<u8> {
        BASE64_ALPHABET.iter().position(|&c| c == byte).map(|i| i as u8)
    }

    let data: Vec<u8> = input
        .bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .take_while(|&b| b != b'=')
        .collect();
    let mut out = Vec::with_capacity(data.len() * 3 / 4 + 3);
    let mut group = [0u8; 4];
    let mut group_len = 0;
    for byte in data {
        group[group_len] = value(byte).ok_or("invalid base64 character")?;
        group_len += 1;
        if group_len == 4 {
            out.push((group[0] << 2) | (group[1] >> 4));
            out.push((group[1] << 4) | (group[2] >> 2));
            out.push((group[2] << 6) | group[3]);
            group_len = 0;
        }
    }
    match group_len {
        0 => {}
        2 => out.push((group[0] << 2) | (group[1] >> 4)),
        3 => {
            out.push((group[0] << 2) | (group[1] >> 4));
            out.push((group[1] << 4) | (group[2] >> 2));
        }
        _ => return Err("truncated base64 data".to_owned()),
    }
    Ok(out)
}

fn decode_data_url(url: &str) -> Result<Vec<u8>, String> {
    let Some(comma) = url.find(',') else {
        return Err(
            "image_url.url must be a data: URI (data:<mime>;base64,<data>) -- remote URLs are not supported yet".into(),
        );
    };
    if !url.starts_with("data:") || !url[..comma].contains("base64") {
        return Err("image_url.url must be a base64 data: URI -- remote URLs are not supported yet".into());
    }
    base64_decode(&url[comma + 1..]).map_err(|err| format!("invalid base64 image data: {err}"))
}

fn build_chat_messages(
    messages: &[ChatMessage],
) -> Result<(Vec<serde_json::Value>, Option<ImagePatches>, Option<AudioFrames>), HttpResponse> {
    let mut image: Option<ImagePatches> = None;
    let mut audio: Option<AudioFrames> = None;
    let mut out = Vec::with_capacity(messages.len());
    for message in messages {
        let content = match &message.content {
            None => serde_json::Value::String(String::new()),
            Some(MessageContent::Text(text)) => serde_json::Value::String(text.clone()),
            Some(MessageContent::Parts(parts)) => {
                let mut image_marker = None;
                let mut text_parts = Vec::with_capacity(parts.len());
                let mut audio_marker = None;
                for part in parts {
                    match part {
                        ContentPart::Text { text } => {
                            text_parts.push(serde_json::json!({ "type": "text", "text": text }));
                        }
                        ContentPart::ImageUrl { image_url } => {
                            if image.is_some() {
                                return Err(error(
                                    400,
                                    "invalid_request_error",
                                    "only one image per request is supported",
                                ));
                            }
                            let bytes = decode_data_url(&image_url.url)
                                .map_err(|msg| error(400, "invalid_request_error", msg))?;
                            let patches = image::load_from_bytes(&bytes).map_err(|err| {
                                error(400, "invalid_request_error", format!("could not decode image: {err}"))
                            })?;
                            image = Some(patches);
                            image_marker = Some(serde_json::json!({ "type": "image" }));
                        }
                        ContentPart::InputAudio { input_audio } => {
                            if audio.is_some() {
                                return Err(error(
                                    400,
                                    "invalid_request_error",
                                    "only one audio input per request is supported",
                                ));
                            }
                            let bytes = base64_decode(&input_audio.data).map_err(|err| {
                                error(
                                    400,
                                    "invalid_request_error",
                                    format!("invalid base64 audio data: {err}"),
                                )
                            })?;
                            let frames = audio::load_from_bytes(&bytes, &input_audio.format).map_err(|err| {
                                error(400, "invalid_request_error", format!("could not decode audio: {err}"))
                            })?;
                            audio = Some(frames);
                            audio_marker = Some(serde_json::json!({ "type": "audio" }));
                        }
                    }
                }
                let json_parts: Vec<_> = image_marker.into_iter().chain(text_parts).chain(audio_marker).collect();
                serde_json::Value::Array(json_parts)
            }
        };
        out.push(serde_json::json!({ "role": message.role, "content": content }));
    }
    Ok((out, image, audio))
}

fn wants_thinking(body: &ChatCompletionRequest) -> Result<bool, HttpResponse> {
    if let Some(kwargs) = &body.chat_template_kwargs {
        let Some(map) = kwargs.as_object() else {
            return Err(error_param(
                400,
                "invalid_request_error",
                "chat_template_kwargs must be an object",
                "chat_template_kwargs",
            ));
        };
        if let Some(value) = map.get("enable_thinking") {
            return value.as_bool().ok_or_else(|| {
                error_param(
                    400,
                    "invalid_request_error",
                    "chat_template_kwargs.enable_thinking must be a boolean",
                    "chat_template_kwargs.enable_thinking",
                )
            });
        }
    }
    match body.reasoning_effort.as_deref() {
        Some("none" | "minimal") => Ok(false),
        Some(_) => Ok(true),
        None => Ok(false),
    }
}

fn sse_error(message: &str) -> Vec<u8> {
    let payload =
        serde_json::to_vec(&ErrorResponse::new("internal_error", message)).expect("error schema always serializes");
    let mut out = Vec::with_capacity(payload.len() + 16);
    out.extend_from_slice(b"event: error\ndata: ");
    out.extend_from_slice(&payload);
    out.extend_from_slice(b"\n\n");
    out
}

fn sse_data<T: Serialize>(value: &T) -> Vec<u8> {
    let payload = serde_json::to_vec(value).expect("chunk schema always serializes");
    let mut out = Vec::with_capacity(payload.len() + 8);
    out.extend_from_slice(b"data: ");
    out.extend_from_slice(&payload);
    out.extend_from_slice(b"\n\n");
    out
}

pub(crate) fn chat_completions(req: &HttpRequest, state: &AppState) -> HttpResponse {
    let body: ChatCompletionRequest = match server::read_json_body(req) {
        Ok(body) => body,
        Err(response) => return response,
    };

    if let Err(response) = validate_model(&body.model) {
        return response;
    }
    if let Err(response) = validate_n(body.n) {
        return response;
    }
    if let Err(response) = validate_top_k(body.top_k) {
        return response;
    }
    if let Err(response) = validate_unsupported(body.unsupported()) {
        return response;
    }
    let wants_audio_output = body.audio.is_some()
        || body
            .modalities
            .as_ref()
            .is_some_and(|modalities| modalities.iter().any(|m| m == "audio"));
    if wants_audio_output {
        return audio_not_implemented();
    }
    if body.messages.is_empty() {
        return error(400, "invalid_request_error", "messages must not be empty");
    }

    let thinking = match wants_thinking(&body) {
        Ok(value) => value,
        Err(response) => return response,
    };

    let (messages, image, audio) = match build_chat_messages(&body.messages) {
        Ok(value) => value,
        Err(response) => return response,
    };
    let chat_ids = match state.tokenizer.encode_chat(&messages, thinking) {
        Ok(ids) => ids,
        Err(err) => {
            return error(
                400,
                "invalid_request_error",
                format!("could not render chat template: {err}"),
            );
        }
    };
    let (prompt_ids, soft_at) = match generate::splice_multimodal_tokens(chat_ids, image.as_ref(), audio.as_ref()) {
        Ok(value) => value,
        Err(message) => return error(400, "invalid_request_error", message),
    };

    let sampling = SamplingConfig {
        temperature: body
            .temperature
            .unwrap_or(SamplingConfig::GEMMA_RECOMMENDED.temperature),
        top_k: body.top_k.unwrap_or(SamplingConfig::GEMMA_RECOMMENDED.top_k),
        top_p: body.top_p.unwrap_or(SamplingConfig::GEMMA_RECOMMENDED.top_p),
    };
    let max_new_tokens = body
        .max_completion_tokens
        .or(body.max_tokens)
        .unwrap_or(generate::DEFAULT_MAX_NEW_TOKENS);
    if let Err(response) = validate_budget(prompt_ids.len(), max_new_tokens, "max_tokens") {
        return response;
    }
    let stop_strings = body.stop.map(OneOrMany::into_vec).unwrap_or_default();
    if let Err(response) = validate_stop(&stop_strings) {
        return response;
    }
    let stream = body.stream;
    let include_usage = body.stream_options.is_some_and(|options| options.include_usage);

    let rx = submit_job(
        state,
        GenerationRequest {
            prompt_ids,
            soft_at,
            image,
            audio,
            max_new_tokens,
            sampling,
            stop_strings,
            separate_reasoning: true,
        },
        stream,
    );

    let id = gen_id("chatcmpl");
    let created = now_secs();

    if stream {
        let (body_tx, body_rx) = mpsc::channel::<Vec<u8>>();
        std::thread::spawn(move || {
            let mut sent_role = false;
            for event in rx {
                let (delta, is_reasoning) = match event {
                    StreamEvent::Delta(delta) => (delta, false),
                    StreamEvent::ReasoningDelta(delta) => (delta, true),
                    StreamEvent::Done(output) => {
                        let usage = Usage::from(&output);
                        let chunk = ChatCompletionChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: MODEL_ID,
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: ChunkDelta::default(),
                                finish_reason: Some(finish_reason_str(output.finish_reason)),
                            }],
                            usage: None,
                        };
                        let _ = body_tx.send(sse_data(&chunk));
                        if include_usage {
                            let chunk = ChatCompletionChunk {
                                id: id.clone(),
                                object: "chat.completion.chunk",
                                created,
                                model: MODEL_ID,
                                choices: Vec::new(),
                                usage: Some(usage),
                            };
                            let _ = body_tx.send(sse_data(&chunk));
                        }
                        let _ = body_tx.send(b"data: [DONE]\n\n".to_vec());
                        break;
                    }
                    StreamEvent::Error(message) => {
                        let _ = body_tx.send(sse_error(&message));
                        let _ = body_tx.send(b"data: [DONE]\n\n".to_vec());
                        break;
                    }
                };
                let chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: MODEL_ID,
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: ChunkDelta {
                            role: if sent_role { None } else { Some("assistant") },
                            content: if is_reasoning { None } else { Some(delta.clone()) },
                            reasoning_content: if is_reasoning { Some(delta) } else { None },
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                };
                sent_role = true;
                if body_tx.send(sse_data(&chunk)).is_err() {
                    break;
                }
            }
        });
        return HttpResponse::event_stream(body_rx);
    }

    block_for_result(rx, |output| {
        let usage = Usage::from(&output);
        json(
            200,
            &ChatCompletionResponse {
                id,
                object: "chat.completion",
                created,
                model: MODEL_ID,
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatCompletionMessage {
                        role: "assistant",
                        reasoning_content: (!output.reasoning.is_empty()).then_some(output.reasoning),
                        content: output.text,
                    },
                    finish_reason: finish_reason_str(output.finish_reason),
                }],
                usage,
            },
        )
    })
}

pub(crate) fn completions(req: &HttpRequest, state: &AppState) -> HttpResponse {
    let body: CompletionRequest = match server::read_json_body(req) {
        Ok(body) => body,
        Err(response) => return response,
    };
    if let Err(response) = validate_model(&body.model) {
        return response;
    }
    if let Err(response) = validate_n(body.n) {
        return response;
    }
    if let Err(response) = validate_unsupported(body.unsupported()) {
        return response;
    }
    let prompts = body.prompt.into_vec();
    if prompts.len() != 1 {
        return error(400, "invalid_request_error", "only a single prompt string is supported");
    }

    let prompt_ids = match state.tokenizer.encode_raw(&prompts[0]) {
        Ok(ids) if !ids.is_empty() => ids,
        Ok(_) => return error(400, "invalid_request_error", "prompt produced no tokens"),
        Err(err) => {
            return error(
                400,
                "invalid_request_error",
                format!("could not tokenize prompt: {err}"),
            );
        }
    };
    let sampling = SamplingConfig {
        temperature: body
            .temperature
            .unwrap_or(SamplingConfig::GEMMA_RECOMMENDED.temperature),
        top_k: SamplingConfig::GEMMA_RECOMMENDED.top_k,
        top_p: body.top_p.unwrap_or(SamplingConfig::GEMMA_RECOMMENDED.top_p),
    };
    let max_new_tokens = body.max_tokens.unwrap_or(generate::DEFAULT_MAX_NEW_TOKENS);
    if let Err(response) = validate_budget(prompt_ids.len(), max_new_tokens, "max_tokens") {
        return response;
    }
    let stop_strings = body.stop.map(OneOrMany::into_vec).unwrap_or_default();
    if let Err(response) = validate_stop(&stop_strings) {
        return response;
    }
    let stream = body.stream;
    let include_usage = body.stream_options.is_some_and(|options| options.include_usage);
    let prompt_len = prompt_ids.len();

    let rx = submit_job(
        state,
        GenerationRequest {
            prompt_ids,
            soft_at: vec![None; prompt_len],
            image: None,
            audio: None,
            max_new_tokens,
            sampling,
            stop_strings,
            separate_reasoning: false,
        },
        stream,
    );

    let id = gen_id("cmpl");
    let created = now_secs();

    if stream {
        let (body_tx, body_rx) = mpsc::channel::<Vec<u8>>();
        std::thread::spawn(move || {
            for event in rx {
                match event {
                    StreamEvent::Delta(delta) | StreamEvent::ReasoningDelta(delta) => {
                        let chunk = CompletionChunk {
                            id: id.clone(),
                            object: "text_completion",
                            created,
                            model: MODEL_ID,
                            choices: vec![CompletionChoice {
                                index: 0,
                                text: delta,
                                finish_reason: None,
                            }],
                            usage: None,
                        };
                        if body_tx.send(sse_data(&chunk)).is_err() {
                            break;
                        }
                    }
                    StreamEvent::Done(output) => {
                        let usage = Usage::from(&output);
                        let chunk = CompletionChunk {
                            id: id.clone(),
                            object: "text_completion",
                            created,
                            model: MODEL_ID,
                            choices: vec![CompletionChoice {
                                index: 0,
                                text: String::new(),
                                finish_reason: Some(finish_reason_str(output.finish_reason)),
                            }],
                            usage: None,
                        };
                        let _ = body_tx.send(sse_data(&chunk));
                        if include_usage {
                            let chunk = CompletionChunk {
                                id: id.clone(),
                                object: "text_completion",
                                created,
                                model: MODEL_ID,
                                choices: Vec::new(),
                                usage: Some(usage),
                            };
                            let _ = body_tx.send(sse_data(&chunk));
                        }
                        let _ = body_tx.send(b"data: [DONE]\n\n".to_vec());
                        break;
                    }
                    StreamEvent::Error(message) => {
                        let _ = body_tx.send(sse_error(&message));
                        let _ = body_tx.send(b"data: [DONE]\n\n".to_vec());
                        break;
                    }
                }
            }
        });
        return HttpResponse::event_stream(body_rx);
    }

    block_for_result(rx, |output| {
        let usage = Usage::from(&output);
        json(
            200,
            &CompletionResponse {
                id,
                object: "text_completion",
                created,
                model: MODEL_ID,
                choices: vec![CompletionChoice {
                    index: 0,
                    text: output.text,
                    finish_reason: Some(finish_reason_str(output.finish_reason)),
                }],
                usage,
            },
        )
    })
}

pub(crate) fn audio_speech(req: &HttpRequest) -> HttpResponse {
    if let Err(response) = server::read_json_body::<AudioSpeechRequest>(req) {
        return response;
    }
    audio_not_implemented()
}

pub(crate) fn audio_transcriptions(_req: &HttpRequest) -> HttpResponse {
    audio_not_implemented()
}

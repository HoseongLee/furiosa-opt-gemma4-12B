
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use furiosa_opt_gemma4::host::generate::{self, Delta, GenerationRequest};
use furiosa_opt_gemma4::host::runtime::Workspace;
use furiosa_opt_gemma4::host::sampling::SamplingConfig;
use furiosa_opt_gemma4::host::tokenizer::Tokenizer;
use furiosa_opt_gemma4::host::{image, load};
use furiosa_opt_std::prelude::*;

const DEFAULT_PROMPT: &str = "Write a hello world program in Rust.";

struct Args {
    image_path: Option<String>,
    prompt: String,
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut image_path = None;
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--image" {
            image_path = Some(args.next().ok_or("--image requires a path")?);
        } else {
            words.push(arg);
        }
    }
    let prompt = if words.is_empty() {
        DEFAULT_PROMPT.to_owned()
    } else {
        words.join(" ")
    };
    Ok(Args { image_path, prompt })
}

fn build_messages(prompt: &str, has_image: bool) -> Vec<serde_json::Value> {
    let content = if has_image {
        serde_json::json!([{ "type": "image" }, { "type": "text", "text": prompt }])
    } else {
        serde_json::json!(prompt)
    };
    vec![serde_json::json!({ "role": "user", "content": content })]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let max_new = std::env::var("GEMMA4_MAX_NEW_TOKENS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(generate::DEFAULT_MAX_NEW_TOKENS);

    let mut ctx = Context::acquire();
    let load_start = Instant::now();
    let model = load::load_model(&mut ctx).await?;
    eprintln!("model loaded in {:.2?}", load_start.elapsed());

    let image = args
        .image_path
        .map(|path| image::load_from_bytes(&std::fs::read(Path::new(&path))?))
        .transpose()?;
    if let Some(patches) = &image {
        eprintln!("image: {} patches", patches.pixels.len());
    }

    let tokenizer = Tokenizer::new(&load::tokenizer_path(&model))?;
    let messages = build_messages(&args.prompt, image.is_some());
    let chat_ids = tokenizer.encode_chat(&messages, false)?;
    let (prompt_ids, soft_at) = generate::splice_multimodal_tokens(chat_ids, image.as_ref(), None)?;

    let mut workspace = Workspace::new(&mut ctx, &model).await;

    println!("{}", args.prompt);
    io::stdout().flush()?;

    let req = GenerationRequest {
        prompt_ids,
        soft_at,
        image,
        audio: None,
        max_new_tokens: max_new,
        sampling: SamplingConfig::GEMMA_RECOMMENDED,
        stop_strings: Vec::new(),
        separate_reasoning: false,
    };

    let run_start = Instant::now();
    let mut first_token_at = None;
    let output = generate::generate(&mut ctx, &model, &tokenizer, &mut workspace, req, |delta| {
        if first_token_at.is_none() {
            first_token_at = Some(Instant::now());
        }
        if let Delta::Content(text) = delta {
            print!("{text}");
            let _ = io::stdout().flush();
        }
        true
    })
    .await?;
    println!();

    let ttft = first_token_at.unwrap_or_else(Instant::now) - run_start;
    let decode_elapsed = run_start.elapsed().saturating_sub(ttft);
    let prefill_tok_per_s = output.prompt_tokens as f64 / ttft.as_secs_f64();
    let decode_tok_per_s = if decode_elapsed.as_secs_f64() > 0.0 {
        output.completion_tokens as f64 / decode_elapsed.as_secs_f64()
    } else {
        0.0
    };
    eprintln!(
        "ttft: {ttft:.2?} ({} prompt tokens, {prefill_tok_per_s:.2} tok/s prefill) | decode: {} tokens in {decode_elapsed:.2?} ({decode_tok_per_s:.2} tok/s) | finish: {:?}",
        output.prompt_tokens, output.completion_tokens, output.finish_reason,
    );
    Ok(())
}

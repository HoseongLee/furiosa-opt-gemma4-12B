
use std::net::SocketAddr;

use furiosa_opt_gemma4::api::server::{self, AppState};
use furiosa_opt_gemma4::api::worker;
use furiosa_opt_gemma4::host::load;
use furiosa_opt_gemma4::host::tokenizer::Tokenizer;

const DEFAULT_ADDR: &str = "0.0.0.0:8000";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = std::env::var("GEMMA4_API_ADDR")
        .unwrap_or_else(|_| DEFAULT_ADDR.to_owned())
        .parse()?;
    let api_key = std::env::var("GEMMA4_API_KEY").ok().filter(|key| !key.is_empty());

    let tokenizer_path = load::tokenizer_path_in(&load::model_dir()?);
    let tokenizer = Tokenizer::new(&tokenizer_path)?;
    let http_tokenizer = tokenizer.clone();

    let worker = worker::spawn(tokenizer)?;

    let state = AppState {
        worker,
        tokenizer: http_tokenizer,
        api_key,
    };
    server::serve(addr, state)?;
    Ok(())
}

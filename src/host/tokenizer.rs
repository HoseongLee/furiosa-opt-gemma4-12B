
use std::path::Path;
use std::sync::Arc;

use llm_tokenizer::chat_template::ChatTemplateParams;
use llm_tokenizer::{DecodeStream, Encoder, HuggingFaceTokenizer, TokenizerTrait};

const BOS_TOKEN: &str = "<bos>";

#[derive(Clone)]
pub struct Tokenizer {
    inner: Arc<HuggingFaceTokenizer>,
}

impl Tokenizer {
    pub fn new(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.to_str().ok_or("tokenizer path is not valid UTF-8")?;
        Ok(Self {
            inner: Arc::new(HuggingFaceTokenizer::from_file(path)?),
        })
    }

    pub fn encode_chat(
        &self,
        messages: &[serde_json::Value],
        thinking: bool,
    ) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        let rendered = self.inner.apply_chat_template(
            messages,
            ChatTemplateParams {
                add_generation_prompt: true,
                thinking: Some(thinking),
                ..Default::default()
            },
        )?;
        Ok(self
            .inner
            .encode(&rendered, false)?
            .token_ids()
            .iter()
            .map(|&id| id as usize)
            .collect())
    }

    pub fn encode_raw(&self, text: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        let with_bos = format!("{BOS_TOKEN}{text}");
        Ok(self
            .inner
            .encode(&with_bos, false)?
            .token_ids()
            .iter()
            .map(|&id| id as usize)
            .collect())
    }

    pub fn decode_stream(&self, prompt_token_ids: &[u32]) -> DecodeStream {
        DecodeStream::new(self.inner.clone(), prompt_token_ids, false)
    }
}

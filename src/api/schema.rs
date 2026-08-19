
use serde::{Deserialize, Serialize};

use crate::host::generate::GenerationOutput;

pub const MODEL_ID: &str = "gemma-4-12b-it";

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

impl<T> OneOrMany<T> {
    pub fn into_vec(self) -> Vec<T> {
        match self {
            Self::One(v) => vec![v],
            Self::Many(v) => v,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct InputAudio {
    pub data: String,
    pub format: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
    InputAudio { input_audio: InputAudio },
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<MessageContent>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub stop: Option<OneOrMany<String>>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub modalities: Option<Vec<String>>,
    #[serde(default)]
    pub audio: Option<serde_json::Value>,
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Value>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,

    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    #[serde(default)]
    pub seed: Option<serde_json::Value>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub logprobs: Option<serde_json::Value>,
    #[serde(default)]
    pub top_logprobs: Option<serde_json::Value>,
    #[serde(default)]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(default)]
    pub frequency_penalty: Option<serde_json::Value>,
    #[serde(default)]
    pub presence_penalty: Option<serde_json::Value>,
}

impl ChatCompletionRequest {
    pub fn unsupported(&self) -> Option<&'static str> {
        first_unsupported(&[
            ("response_format", &self.response_format),
            ("seed", &self.seed),
            ("tools", &self.tools),
            ("tool_choice", &self.tool_choice),
            ("logprobs", &self.logprobs),
            ("top_logprobs", &self.top_logprobs),
            ("logit_bias", &self.logit_bias),
            ("frequency_penalty", &self.frequency_penalty),
            ("presence_penalty", &self.presence_penalty),
        ])
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

fn asks_for_nothing(value: &serde_json::Value) -> bool {
    value.is_null()
        || value == &serde_json::Value::Bool(false)
        || value.as_f64().is_some_and(|number| number == 0.0)
        || value.as_str().is_some_and(|text| text == "none" || text == "auto")
        || value.as_array().is_some_and(|items| items.is_empty())
        || value.get("type").and_then(serde_json::Value::as_str) == Some("text")
        || value.as_object().is_some_and(|fields| fields.is_empty())
}

fn first_unsupported(fields: &[(&'static str, &Option<serde_json::Value>)]) -> Option<&'static str> {
    fields
        .iter()
        .find(|(_, value)| value.as_ref().is_some_and(|value| !asks_for_nothing(value)))
        .map(|(name, _)| *name)
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl From<&GenerationOutput> for Usage {
    fn from(output: &GenerationOutput) -> Self {
        Self {
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: output.prompt_tokens + output.completion_tokens,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionMessage {
    pub role: &'static str,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatCompletionMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: &'static str,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize, Default)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: ChunkDelta,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: &'static str,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: OneOrMany<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop: Option<OneOrMany<String>>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,

    #[serde(default)]
    pub seed: Option<serde_json::Value>,
    #[serde(default)]
    pub logprobs: Option<serde_json::Value>,
    #[serde(default)]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(default)]
    pub frequency_penalty: Option<serde_json::Value>,
    #[serde(default)]
    pub presence_penalty: Option<serde_json::Value>,
}

impl CompletionRequest {
    pub fn unsupported(&self) -> Option<&'static str> {
        first_unsupported(&[
            ("seed", &self.seed),
            ("logprobs", &self.logprobs),
            ("logit_bias", &self.logit_bias),
            ("frequency_penalty", &self.frequency_penalty),
            ("presence_penalty", &self.presence_penalty),
        ])
    }
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: &'static str,
    pub choices: Vec<CompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: &'static str,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: &'static str,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Serialize)]
pub struct ModelsList {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Deserialize)]
pub struct AudioSpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    #[serde(default)]
    pub response_format: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: &'static str,
    pub param: Option<&'static str>,
    pub code: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

impl ErrorResponse {
    pub fn new(error_type: &'static str, message: impl Into<String>) -> Self {
        Self {
            error: ErrorBody {
                message: message.into(),
                error_type,
                param: None,
                code: None,
            },
        }
    }

    pub fn with_param(error_type: &'static str, message: impl Into<String>, param: &'static str) -> Self {
        Self {
            error: ErrorBody {
                message: message.into(),
                error_type,
                param: Some(param),
                code: None,
            },
        }
    }
}
